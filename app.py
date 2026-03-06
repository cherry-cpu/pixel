from __future__ import annotations

import io
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image

from backend.photo_service import (
    all_people,
    get_photo_path,
    init_app_storage,
    list_photos,
    save_photo,
    search_photos_by_text,
    share_photos,
    update_photo_labels,
)


def _init() -> None:
    st.set_page_config(page_title="AI Photo Manager", layout="wide")
    init_app_storage()


def sidebar() -> str:
    st.sidebar.title("AI Photo Manager")
    page = st.sidebar.radio(
        "Navigation",
        ["Upload & Organize", "People", "Search", "Share"],
        index=0,
    )
    with st.sidebar.expander("About", expanded=False):
        st.write(
            "This app uses HuggingFace embeddings + Streamlit to organize, search, "
            "and share your photos with natural language and face-aware grouping."
        )
    return page


def page_upload() -> None:
    st.header("Upload & Organize")
    uploaded_files = st.file_uploader(
        "Upload photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        default_person = st.text_input("Default person label (optional)")
    with col2:
        default_tags = st.text_input("Default tags (comma separated)")
    with col3:
        is_private = st.checkbox("Mark as private by default", value=True)

    if uploaded_files and st.button("Save photos with AI organization"):
        tags_list: List[str] = []
        if default_tags.strip():
            tags_list = [t.strip() for t in default_tags.split(",") if t.strip()]

        saved = []
        with st.spinner("Processing photos with HuggingFace embeddings..."):
            for f in uploaded_files:
                bytes_data = f.read()
                photo = save_photo(
                    bytes_data,
                    original_name=f.name,
                    person_label=default_person or None,
                    tags=tags_list or None,
                    caption=None,
                    is_private=is_private,
                )
                saved.append(photo)

        st.success(f"Saved {len(saved)} photos.")

    st.subheader("Recent photos")
    photos = list_photos()
    if not photos:
        st.info("No photos yet. Upload some to get started.")
        return
    cols = st.columns(4)
    for idx, p in enumerate(photos[:12]):
        img_path = get_photo_path(p)
        if not img_path.exists():
            continue
        with cols[idx % 4]:
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
            st.caption(
                f"{p.original_name}\n\n"
                f"Person: {p.person_label or 'Unknown'} | "
                f"Private: {'Yes' if p.is_private else 'No'}"
            )


def page_people() -> None:
    st.header("People (Face-aware organization)")
    people = all_people()
    cols = st.columns([2, 1])
    with cols[0]:
        selected_person = st.selectbox(
            "Select person label", options=["<All>"] + people, index=0
        )
    with cols[1]:
        new_person = st.text_input("Create / rename label")

    target_person = None if selected_person == "<All>" else selected_person
    photos = list_photos(person=target_person)

    if not photos:
        st.info("No photos for this person yet.")
        return

    st.write(
        "Select photos to assign or update person labels, privacy, and tags. "
        "Note: Face recognition here is powered by image embeddings; you define names."
    )

    selected_ids: List[int] = []
    cols = st.columns(4)
    for idx, p in enumerate(photos):
        img_path = get_photo_path(p)
        if not img_path.exists():
            continue
        with cols[idx % 4]:
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
            st.caption(
                f"{p.original_name}\nPerson: {p.person_label or 'Unknown'}\n"
                f"Private: {'Yes' if p.is_private else 'No'}"
            )
            if st.checkbox(f"Select #{p.id}", key=f"sel_{p.id}"):
                selected_ids.append(p.id)

    st.markdown("---")
    st.subheader("Bulk update selected photos")
    col1, col2, col3 = st.columns(3)
    with col1:
        assign_person = st.text_input("Assign person label to selected", value=new_person)
    with col2:
        assign_tags = st.text_input("Assign tags (comma separated)")
    with col3:
        make_private = st.selectbox(
            "Privacy for selected",
            options=["No change", "Private", "Shareable"],
            index=0,
        )

    if st.button("Apply updates to selected photos"):
        if not selected_ids:
            st.warning("No photos selected.")
        else:
            tags_list = (
                [t.strip() for t in assign_tags.split(",") if t.strip()]
                if assign_tags.strip()
                else None
            )
            is_private = None
            if make_private == "Private":
                is_private = True
            elif make_private == "Shareable":
                is_private = False

            update_photo_labels(
                selected_ids,
                person_label=assign_person or None,
                tags=tags_list,
                is_private=is_private,
            )
            st.success("Updated selected photos.")


def page_search() -> None:
    st.header("Natural Language Search")
    query = st.text_input(
        "Describe what you're looking for",
        placeholder="e.g. me and John at the beach at sunset",
    )
    top_k = st.slider("Max results", min_value=4, max_value=40, value=16, step=4)

    if st.button("Search") and query.strip():
        with st.spinner("Searching photos using CLIP embeddings..."):
            results = search_photos_by_text(query.strip(), top_k=top_k)

        if not results:
            st.info("No matching photos found.")
            return

        cols = st.columns(4)
        for idx, (p, score) in enumerate(results):
            img_path = get_photo_path(p)
            if not img_path.exists():
                continue
            with cols[idx % 4]:
                img = Image.open(img_path)
                st.image(img, use_column_width=True)
                st.caption(
                    f"{p.original_name}\n"
                    f"Person: {p.person_label or 'Unknown'} | "
                    f"Score: {score:.3f}"
                )


def page_share() -> None:
    st.header("Automated Sharing")
    st.write(
        "Select non-private photos to bundle into a shareable folder. "
        "Private photos are never copied into share bundles."
    )

    photos = list_photos()
    if not photos:
        st.info("No photos available.")
        return

    selected_ids: List[int] = []
    cols = st.columns(4)
    for idx, p in enumerate(photos):
        if p.is_private:
            continue
        img_path = get_photo_path(p)
        if not img_path.exists():
            continue
        with cols[idx % 4]:
            img = Image.open(img_path)
            st.image(img, use_column_width=True)
            st.caption(
                f"{p.original_name}\nPerson: {p.person_label or 'Unknown'}"
            )
            if st.checkbox(f"Select share #{p.id}", key=f"share_{p.id}"):
                selected_ids.append(p.id)

    if st.button("Create share bundle"):
        if not selected_ids:
            st.warning("No photos selected.")
            return
        share_path = share_photos(selected_ids)
        st.success("Share bundle created.")
        st.code(str(share_path), language="bash")
        st.write(
            "You can compress this folder (zip) and send it via your preferred channel. "
            "For security, photos marked as private were excluded."
        )


def main() -> None:
    _init()
    page = sidebar()
    if page == "Upload & Organize":
        page_upload()
    elif page == "People":
        page_people()
    elif page == "Search":
        page_search()
    elif page == "Share":
        page_share()


if __name__ == "__main__":
    main()

