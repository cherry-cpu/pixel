from __future__ import annotations
from random import randint
import io
import json
from dataclasses import replace

import streamlit as st
import zipfile
from PIL import Image

from pixel_app.core.llm import ParsedQuery, parse_query_with_llm


def _require_unlocked(app) -> bool:
    if app.auth.crypto is None:
        st.warning("Library is locked. Enter your passphrase in the sidebar.")
        return False
    return True


def _photo_card(app, photo_row: dict) -> None:
    col1, col2 = st.columns([1, 2], vertical_alignment="top")
    with col1:
        try:
            thumb = app.library.read_thumbnail_bytes(photo_row)
            st.image(thumb, use_container_width=True)
        except Exception:
            st.info("Thumbnail unavailable.")
    with col2:
        st.markdown(f"**{photo_row['original_name']}**")
        st.caption(photo_row["added_at"])
        user_tags = app.search.get_photo_tags(photo_row)
        auto_tags = app.search.get_photo_auto_tags(photo_row)
        auto_caption = app.search.get_photo_auto_caption(photo_row)
        st.write(
            {
                "auto_caption": auto_caption,
                "auto_tags": [t for t in auto_tags if ":" not in t],
                "caption": photo_row.get("caption") or "",
                "tags": user_tags,
            }
        )

        new_caption = st.text_input(
            "Caption",
            value=photo_row.get("caption") or "",
            key=f"cap_{photo_row['id']}_{randint(0,100)}",
        )
        if new_caption != (photo_row.get("caption") or ""):
            app.search.set_photo_caption(photo_row["id"], new_caption)
            st.rerun()

        tags_str = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(user_tags),
            key=f"tags_{photo_row['id']}_{randint(0,100)}",
        )
        new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        if new_tags != user_tags:
            app.search.set_photo_tags(photo_row["id"], new_tags)
            st.rerun()

        faces = app.library.get_photo_faces(photo_row["id"])
        if faces:
            st.write("Faces:")
            people_list = app.people.list_people()
            person_options = {p["name"]: p["id"] for p in people_list}
            opts = ["(unassigned)"] + list(person_options.keys())
            for f in faces:
                face_id = f.get("id")
                person_name = f.get("person_name") or "(unassigned)"
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    conf = f.get("confidence")
                    st.caption(f"{person_name}" + (f" (confidence: {conf:.2f})" if conf is not None else ""))
                with col_b:
                    if face_id:
                        current_id = f.get("person_id")
                        current_name = next((n for n, pid in person_options.items() if pid == current_id), None)
                        idx = 0 if not current_name or current_name not in opts else opts.index(current_name)
                        chosen = st.selectbox(
                            "Assign to",
                            options=opts,
                            index=min(idx, len(opts) - 1),
                            key=f"face_assign_{photo_row['id']}_{face_id}",
                        )
                        new_id = None if chosen == "(unassigned)" else person_options.get(chosen)
                        if new_id != current_id:
                            app.people.assign_face_to_person(face_id, new_id)
                            st.rerun()
            st.caption("Assign faces to people above; use 'Cluster unknown faces' in Library to group similar faces.")


def page_library(app) -> None:
    st.header("Library")

    if not _require_unlocked(app):
        return

    st.subheader("Add photos")
    uploads = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        added = 0
        skipped = 0
        for up in uploads:
            ok, _msg = app.library.ingest(up.name, up.read())
            if ok:
                added += 1
            else:
                skipped += 1
        st.success(f"Done. Added {added}, skipped {skipped}.")

    st.divider()
    st.subheader("Auto-organize (faces)")
    if st.button("Cluster unknown faces"):
        stats = app.people.auto_cluster_unknown_faces()
        st.success(
            f"Assigned {stats['assigned_faces']} faces; created {stats['created_people']} people."
        )

    st.divider()
    st.subheader("Explore")
    tab_grid, tab_timeline, tab_quality = st.tabs(["Recent grid", "Timeline & events", "Quality & duplicates"])

    with tab_grid:
        rows = app.library.list_photos(limit=60, offset=0)
        if not rows:
            st.info("No photos yet. Upload a few above.")
        else:
            for r in rows:
                with st.container(border=True):
                    _photo_card(app, r)

    with tab_timeline:
        st.caption("Photos grouped into events based on when they were taken.")
        events = app.search.build_events(gap_hours=3.0)
        if not events:
            st.info("No events yet. Add some photos with EXIF dates.")
        else:
            for ev in events:
                with st.container(border=True):
                    st.markdown(
                        f"**Event {ev['id']}** — {ev['start_iso'][:10]} to {ev['end_iso'][:10]} "
                        f"({ev['count']} photos)"
                    )
                    # show a few thumbnails inline
                    cols = st.columns(min(4, len(ev["photo_ids"])))
                    for idx, pid in enumerate(ev["photo_ids"][:4]):
                        photo = app.library.get_photo(pid)
                        if not photo:
                            continue
                        with cols[idx]:
                            try:
                                thumb = app.library.read_thumbnail_bytes(photo)
                                st.image(thumb, use_container_width=False)
                            except Exception:
                                st.text(photo["original_name"])

    with tab_quality:
        st.caption("See top-quality photos and near-duplicate groups.")
        top = app.search.get_top_quality(limit=20)
        if top:
            st.markdown("**Top picks (by sharpness/contrast):**")
            for r in top:
                with st.container(border=True):
                    _photo_card(app, r)
        else:
            st.info("No photos scored yet.")

        st.markdown("---")
        st.markdown("**Near-duplicate groups:**")
        groups = app.search.find_duplicate_groups(max_hamming=4, min_group_size=2)
        if not groups:
            st.info("No near-duplicate groups detected yet.")
        else:
            for g in groups:
                with st.container(border=True):
                    st.caption(f"Group phash={g['phash']} ({len(g['photo_ids'])} photos)")
                    for pid in g["photo_ids"]:
                        photo = app.library.get_photo(pid)
                        if not photo:
                            continue
                        _photo_card(app, photo)

'''

def page_people(app) -> None:
    st.header("People")
    if not _require_unlocked(app):
        return

    st.subheader("Known people")
    people = app.people.list_people()
    if not people:
        st.info("No people yet. Upload photos and run clustering in Library.")
        return

    for p in people:
        with st.container(border=True):
            c1, c2 = st.columns([2, 1], vertical_alignment="center")
            with c1:
                st.markdown(f"**{p['name']}**")
                st.caption(p["id"])
            with c2:
                new_name = st.text_input("Rename", value=p["name"], key=f"rename_{p['id']}")
                if new_name.strip() and new_name.strip() != p["name"]:
                    app.people.rename_person(p["id"], new_name.strip())
                    st.rerun()


'''
def page_search(app) -> None:
    st.header("Search")
    if not _require_unlocked(app):
        return

    people_list = app.people.list_people()
    person_names = [p["name"] for p in people_list]
    quick_person = st.selectbox(
        "Quick filter: photos of person",
        options=["(any)"] + person_names,
        index=0,
        help="Filter by recognized person without typing.",
    )

    q = st.text_input(
        "Or search by natural language / keywords",
        placeholder="e.g. 'photos with Arjun at the beach' or 'last week sunset'",
    )
    use_llm = st.toggle("Use natural language (LLM)", value=True)

    parsed = None
    if q and use_llm:
        parsed = parse_query_with_llm(q)

    if quick_person and quick_person != "(any)":
        # Combine with NL result if both set; otherwise just filter by person
        if parsed is not None:
            merged_people = list(set(parsed.people) | {quick_person})
            parsed = replace(parsed, people=merged_people)
        else:
            parsed = ParsedQuery(
                people=[quick_person],
                tags=[],
                text="",
                date_from=None,
                date_to=None,
                limit=60,
            )

    if parsed is not None:
        with st.expander("Interpreted query (LLM)", expanded=False):
            st.json(parsed.__dict__)
        rows = app.search.structured_search(parsed)
    else:
        rows = app.search.keyword_search(q, limit=60)

    st.write(f"Results: {len(rows)}")
    for r in rows:
        with st.container(border=True):
            _photo_card(app, r)


def page_share(app) -> None:
    st.header("Share")
    if not _require_unlocked(app):
        return

    st.subheader("Create a share package")
    photos = app.library.list_photos(limit=200, offset=0)
    options = {f"{p['original_name']} — {p['id'][:8]}": p["id"] for p in photos}
    selected = st.multiselect("Select photos", options=list(options.keys()))
    note = st.text_input("Note (optional)")

    if st.button("Create package", disabled=(len(selected) == 0)):
        ids = [options[k] for k in selected]
        out = app.sharing.create_share_package(ids, note=note or None)
        st.success("Share package created.")
        st.code(out["token"], language="text")
        st.caption("This token decrypts the package. Treat it like a password.")
        st.download_button(
            "Download token file (JSON)",
            data=out["download_bytes"],
            file_name="pixel_share_token.json",
            mime="application/json",
        )
        st.info(f"Saved encrypted payload at: {out['payload_path']}")

    st.divider()
    st.subheader("Import & view a share package (read-only)")
    token = st.text_input("Share token", type="password", help="Token from the sender.")
    payload = st.file_uploader("Upload the .share payload", type=["share"])

    if token and payload and st.button("Open package"):
        try:
            zip_bytes = app.sharing.decrypt_share_payload(token.strip(), payload.read())
            with st.spinner("Decrypting & loading..."):
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
                    manifest = json.loads(z.read("manifest.json").decode("utf-8"))
                    st.json(manifest)
                    for item in manifest.get("photos", []):
                        pid = item["id"]
                        name = item["original_name"]
                        arc = next((n for n in z.namelist() if n.startswith(pid + "_")), None)
                        if not arc:
                            continue
                        raw = z.read(arc)
                        img = Image.open(io.BytesIO(raw))
                        st.image(img, caption=name, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to open package: {e}")


def page_settings(app) -> None:
    st.header("Settings")

    st.subheader("Storage")
    st.write(
        {
            "data_dir": str(app.paths.data_dir),
            "db_path": str(app.paths.db_path),
            "photos_dir": str(app.paths.photos_dir),
            "thumbs_dir": str(app.paths.thumbs_dir),
            "shares_dir": str(app.paths.shares_dir),
        }
    )

    st.subheader("Maintenance")
    if st.button("Delete cached thumbnails"):
        deleted = 0
        if app.paths.thumbs_dir.exists():
            for p in app.paths.thumbs_dir.glob("*.jpg"):
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    pass
        st.success(f"Deleted {deleted} thumbnails.")

