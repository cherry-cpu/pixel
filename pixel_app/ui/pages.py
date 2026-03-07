from __future__ import annotations

import io
import json

import streamlit as st
import zipfile
from PIL import Image

from pixel_app.core.llm import parse_query_with_llm





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
        tags = app.search.get_photo_tags(photo_row)
        # Display Caption
        if photo_row.get("caption"):
            st.markdown(f"*{photo_row['caption']}*")
            
        # Display Tags as Visual Pill Badges (Matching Reference Video)
        if tags:
            tag_html = ""
            for tag in tags:
                tag_html += f'<span style="display:inline-block; background-color:#1E293B; color:#E2E8F0; border-radius:12px; padding:2px 10px; margin:2px 4px 2px 0; font-size:12px; font-weight:600;">#{tag}</span>'
            st.markdown(tag_html, unsafe_allow_html=True)

        new_caption = st.text_input(
            "Caption",
            value=photo_row.get("caption") or "",
            key=f"cap_{photo_row['id']}",
        )
        if new_caption != (photo_row.get("caption") or ""):
            app.search.set_photo_caption(photo_row["id"], new_caption)
            st.rerun()

        tags_str = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(tags),
            key=f"tags_{photo_row['id']}",
        )
        new_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        if new_tags != tags:
            app.search.set_photo_tags(photo_row["id"], new_tags)
            st.rerun()

        faces = app.library.get_photo_faces(photo_row["id"])
        if faces:
            st.write(
                "Faces:",
                [
                    (f.get("person_name") or "(unassigned)", f.get("confidence"))
                    for f in faces
                ],
            )


def page_dashboard(app) -> None:
    st.header("Dashboard")
    st.caption("Overview of your AI-Organized Library")
    
    # Calculate Metrics
    photos = app.library.list_photos(limit=10000, offset=0)
    total_photos = len(photos)
    
    people = app.people.list_people()
    total_people = len(people)
    
    # Extract unique background tags
    all_tags = set()
    for p in photos:
        for t in app.search.get_photo_tags(p):
            all_tags.add(t)
            
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Photos", total_photos)
    col2.metric("People Identified", total_people)
    col3.metric("AI Background Tags", len(all_tags))
    
    st.divider()
    st.subheader("Recently Added")
    
    rows = app.library.list_photos(limit=3, offset=0)
    if rows:
        cols = st.columns(3)
        for i, r in enumerate(rows):
            with cols[i % 3]:
                try:
                    thumb = app.library.read_thumbnail_bytes(r)
                    st.image(thumb, use_container_width=True)
                    st.caption(f"{r['original_name']}")
                    tags = app.search.get_photo_tags(r)
                    if tags:
                        tag_html = ""
                        for tag in tags[:3]: # limit to 3 tags on dashboard so it doesnt overflow
                            tag_html += f'<span style="display:inline-block; background-color:#1E293B; color:#E2E8F0; border-radius:12px; padding:2px 8px; margin:2px 2px 2px 0; font-size:10px; font-weight:600;">#{tag}</span>'
                        st.markdown(tag_html, unsafe_allow_html=True)
                except:
                    pass
    else:
        st.info("No photos uploaded yet.")


def page_library(app) -> None:
    st.header("Library")

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
    # st.subheader("Auto-organize (faces)")
    # c1, c2 = st.columns([1, 2])
    #with c1:
    #    thr = st.slider("Similarity threshold", 0.60, 0.90, 0.78, 0.01)
    #with c2:
    #    if st.button("Cluster unknown faces"):
    #        stats = app.people.auto_cluster_unknown_faces(sim_threshold=float(thr))
    #        st.success(f"Assigned {stats['assigned_faces']} faces; created {stats['created_people']} people.")
    #st.divider()
    st.subheader("Recent photos")
    rows = app.library.list_photos(limit=60, offset=0)
    if not rows:
        st.info("No photos yet. Upload a few above.")
        return

    for r in rows:
        with st.container(border=True):
            _photo_card(app, r)


def page_people(app) -> None:
    st.header("People")

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


def page_search(app) -> None:
    st.header("Search")

    q = st.text_input(
        "Search your library",
        placeholder="e.g. 'photos with Arjun at the beach' or 'last week sunset'",
    )
    use_llm = st.toggle("Use natural language (LLM)", value=True)

    parsed = None
    if q and use_llm:
        parsed = parse_query_with_llm(q)

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


def page_chatbot(app) -> None:
    st.header("Drishyamitra Assistant")
    st.caption("Ask me anything about your photos!")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg:
                cols = st.columns(min(3, len(msg["images"])))
                for i, r in enumerate(msg["images"]):
                    with cols[i % 3]:
                        try:
                            thumb = app.library.read_thumbnail_bytes(r)
                            st.image(thumb, use_container_width=True)
                        except:
                            pass

    user_input = st.chat_input("Show me photos of Rohan at the beach...")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                parsed = parse_query_with_llm(user_input)
                if parsed:
                    # Execute search
                    rows = app.search.structured_search(parsed)
                    
                    if rows:
                        reply = f"I found {len(rows)} photos matching your request!"
                        st.markdown(reply)
                        cols = st.columns(min(3, len(rows)))
                        for i, r in enumerate(rows):
                            with cols[i % 3]:
                                try:
                                    thumb = app.library.read_thumbnail_bytes(r)
                                    st.image(thumb, use_container_width=True)
                                except:
                                    pass
                        st.session_state["chat_history"].append({
                            "role": "assistant",
                            "content": reply,
                            "images": rows[:3] # Show up to 3 thumbnails in history
                        })
                    else:
                        reply = "I couldn't find any photos perfectly matching that description."
                        st.markdown(reply)
                        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                else:
                    reply = "I'm having trouble understanding exactly what photos you are looking for."
                    st.markdown(reply)
                    st.session_state["chat_history"].append({"role": "assistant", "content": reply})


def page_share(app) -> None:
    st.header("Share")

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

