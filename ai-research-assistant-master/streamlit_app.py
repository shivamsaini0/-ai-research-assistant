"""Streamlit frontend for the AI Research Assistant."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")
HEALTH_URL = os.getenv("HEALTH_URL", "http://localhost:8000/health")


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)


def api_client() -> httpx.Client:
    token = st.session_state.get("auth_token")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(timeout=120.0, headers=headers)


def backend_healthy(client: httpx.Client) -> bool:
    try:
        response = client.get(HEALTH_URL)
        return response.is_success
    except httpx.HTTPError:
        return False


def _api_error_message(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        try:
            payload = error.response.json()
            return payload.get("detail", error.response.text)
        except Exception:
            return error.response.text or str(error)
    return str(error)


def register_user(client: httpx.Client, name: str, email: str, password: str) -> dict[str, Any]:
    response = client.post(
        f"{API_BASE_URL}/auth/register",
        json={"name": name, "email": email, "password": password},
    )
    response.raise_for_status()
    return response.json()


def login_user(client: httpx.Client, email: str, password: str) -> dict[str, Any]:
    response = client.post(
        f"{API_BASE_URL}/auth/login",
        json={"email": email, "password": password},
    )
    response.raise_for_status()
    return response.json()


def get_me(client: httpx.Client) -> dict[str, Any]:
    response = client.get(f"{API_BASE_URL}/auth/me")
    response.raise_for_status()
    return response.json()


def load_documents(client: httpx.Client) -> list[dict[str, Any]]:
    response = client.get(f"{API_BASE_URL}/documents")
    response.raise_for_status()
    payload = response.json()
    return payload.get("documents", [])


def upload_file(client: httpx.Client, file) -> dict[str, Any]:
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    response = client.post(f"{API_BASE_URL}/upload", files=files)
    response.raise_for_status()
    return response.json()


def delete_document(client: httpx.Client, doc_id: str) -> None:
    response = client.delete(f"{API_BASE_URL}/documents/{doc_id}")
    response.raise_for_status()


def clear_session(client: httpx.Client, session_id: str) -> None:
    response = client.delete(f"{API_BASE_URL}/chat/{session_id}")
    response.raise_for_status()


def send_chat(client: httpx.Client, session_id: str, message: str, document_ids: list[str] | None) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "message": message,
        "document_ids": document_ids or None,
    }
    response = client.post(f"{API_BASE_URL}/chat", json=payload)
    response.raise_for_status()
    return response.json()


def quick_eval(client: httpx.Client, document_ids: list[str] | None) -> dict[str, Any]:
    response = client.post(
        f"{API_BASE_URL}/evaluate/quick",
        json={"document_ids": document_ids or None},
    )
    response.raise_for_status()
    return response.json()


def custom_eval(client: httpx.Client, samples: list[dict[str, Any]], document_ids: list[str] | None) -> dict[str, Any]:
    response = client.post(
        f"{API_BASE_URL}/evaluate",
        json={
            "samples": samples,
            "session_id": st.session_state["session_id"],
            "document_ids": document_ids or None,
        },
    )
    response.raise_for_status()
    return response.json()


def init_state() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("documents", [])
    st.session_state.setdefault("selected_doc_ids", [])
    st.session_state.setdefault("auth_token", None)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("auth_mode", "Login")
    st.session_state.setdefault(
        "samples",
        [
            {"question": "What is the main topic of the uploaded documents?", "ground_truth": ""},
        ],
    )


def render_sidebar(client: httpx.Client) -> None:
    st.sidebar.title("ResearchAI")
    st.sidebar.caption("Streamlit UI for the RAG backend")

    if backend_healthy(client):
        st.sidebar.success("Backend connected")
    else:
        st.sidebar.error("Backend unreachable")

    st.sidebar.write("Session")
    st.sidebar.code(st.session_state["session_id"], language="text")

    if st.session_state.get("current_user"):
        user = st.session_state["current_user"]
        st.sidebar.success(f"Signed in as {user['name']}")
        st.sidebar.write("User profile")
        st.sidebar.json(
            {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
            },
        )
        if st.sidebar.button("Logout"):
            st.session_state["auth_token"] = None
            st.session_state["current_user"] = None
            st.session_state["messages"] = []
            st.session_state["documents"] = []
            st.session_state["selected_doc_ids"] = []
            st.session_state["session_id"] = str(uuid.uuid4())
            st.rerun()
        return

    st.sidebar.write("Account")
    st.session_state["auth_mode"] = st.sidebar.selectbox("Mode", ["Login", "Register"], index=0 if st.session_state["auth_mode"] == "Login" else 1)

    with st.sidebar.form("auth_form"):
        if st.session_state["auth_mode"] == "Register":
            name = st.text_input("Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create account")
            if submitted:
                try:
                    result = register_user(client, name, email, password)
                    st.session_state["auth_token"] = result["access_token"]
                    st.session_state["current_user"] = result["user"]
                    st.session_state["messages"] = []
                    st.session_state["documents"] = []
                    st.session_state["selected_doc_ids"] = []
                    st.session_state["session_id"] = str(uuid.uuid4())
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(_api_error_message(exc))
        else:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
            if submitted:
                try:
                    result = login_user(client, email, password)
                    st.session_state["auth_token"] = result["access_token"]
                    st.session_state["current_user"] = result["user"]
                    try:
                        st.session_state["current_user"] = get_me(api_client())
                    except Exception:
                        pass
                    st.session_state["messages"] = []
                    st.session_state["documents"] = []
                    st.session_state["selected_doc_ids"] = []
                    st.session_state["session_id"] = str(uuid.uuid4())
                    st.rerun()
                except Exception as exc:
                    st.sidebar.error(_api_error_message(exc))

    if st.sidebar.button("New chat session"):
        try:
            clear_session(client, st.session_state["session_id"])
        except httpx.HTTPError:
            pass
        st.session_state["session_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.rerun()


def render_upload_panel(client: httpx.Client) -> None:
    if not st.session_state.get("auth_token"):
        st.info("Sign in to upload documents.")
        return
    st.subheader("Upload documents")
    files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Upload to knowledge base", disabled=not files):
        for file in files or []:
            with st.spinner(f"Uploading {file.name}..."):
                result = upload_file(client, file)
                st.success(f"Indexed {file.name} into {result['total_chunks']} chunks")
        st.session_state["documents"] = load_documents(client)
        st.rerun()


def render_documents_panel(client: httpx.Client) -> None:
    if not st.session_state.get("auth_token"):
        st.info("Sign in to view documents.")
        return
    st.subheader("Documents")
    documents = load_documents(client)
    st.session_state["documents"] = documents

    if not documents:
        st.info("No documents uploaded yet.")
        return

    options = {
        f"{doc['filename']} ({doc['total_chunks']} chunks)": doc["doc_id"]
        for doc in documents
    }

    selected_labels = st.multiselect(
        "Filter chat/evaluation to selected documents",
        options=list(options.keys()),
        default=[label for label, doc_id in options.items() if doc_id in st.session_state["selected_doc_ids"]],
    )
    st.session_state["selected_doc_ids"] = [options[label] for label in selected_labels]

    for doc in documents:
        cols = st.columns([4, 1, 1])
        cols[0].write(f"**{doc['filename']}**  ")
        cols[0].caption(f"{doc['total_chunks']} chunks · {doc['file_type']} · {doc['status']}")
        cols[1].write(doc["doc_id"][:8])
        if cols[2].button("Delete", key=f"delete-{doc['doc_id']}"):
            delete_document(client, doc["doc_id"])
            st.rerun()


def render_chat_panel(client: httpx.Client) -> None:
    if not st.session_state.get("auth_token"):
        st.info("Sign in to chat with your documents.")
        return
    st.subheader("Chat")
    st.caption("Answers are grounded in retrieved document chunks and keep conversation history per session.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat(
                    client,
                    st.session_state["session_id"],
                    prompt,
                    st.session_state["selected_doc_ids"],
                )
                answer = response["answer"]
                st.markdown(answer)

                if response.get("sources"):
                    with st.expander(f"Sources ({len(response['sources'])})", expanded=False):
                        for index, source in enumerate(response["sources"], start=1):
                            st.write(f"{index}. **{source['filename']}** - chunk {source['chunk_index']}")
                            st.caption(source["content_preview"])

        st.session_state["messages"].append({"role": "assistant", "content": answer})


def render_eval_panel(client: httpx.Client) -> None:
    if not st.session_state.get("auth_token"):
        st.info("Sign in to run evaluations.")
        return
    st.subheader("Evaluation")
    st.caption("Run a quick system check or custom question set against selected documents.")

    col1, col2 = st.columns(2)
    if col1.button("Run quick evaluation"):
        with st.spinner("Running quick evaluation..."):
            result = quick_eval(client, st.session_state["selected_doc_ids"])
            st.json(result)

    if col2.button("Reset evaluation samples"):
        st.session_state["samples"] = [{"question": "What is the main topic of the uploaded documents?", "ground_truth": ""}]

    st.write("Custom samples")
    samples = st.session_state["samples"]
    for index, sample in enumerate(samples):
        st.markdown(f"**Sample {index + 1}**")
        sample["question"] = st.text_input(
            "Question",
            value=sample.get("question", ""),
            key=f"sample-question-{index}",
        )
        sample["ground_truth"] = st.text_area(
            "Ground truth (optional)",
            value=sample.get("ground_truth", ""),
            key=f"sample-truth-{index}",
            height=90,
        )

    c1, c2 = st.columns(2)
    if c1.button("Add sample"):
        samples.append({"question": "", "ground_truth": ""})
        st.rerun()

    if c2.button("Run custom evaluation"):
        valid_samples = [s for s in samples if s.get("question", "").strip()]
        if not valid_samples:
            st.warning("Add at least one question before running evaluation.")
        else:
            with st.spinner("Running evaluation..."):
                result = custom_eval(client, valid_samples, st.session_state["selected_doc_ids"])
                st.json(result)


def main() -> None:
    init_state()
    public_client = httpx.Client(timeout=120.0)

    st.title("AI Research Assistant")
    st.write("Upload documents, chat with grounded answers, and run RAG evaluation from one Streamlit interface.")

    render_sidebar(public_client)

    if not st.session_state.get("auth_token"):
        st.info("Create an account or sign in from the sidebar to use the assistant.")
        return

    client = api_client()

    tab_upload, tab_chat, tab_docs, tab_eval = st.tabs(["Upload", "Chat", "Documents", "Evaluation"])

    with tab_upload:
        render_upload_panel(client)

    with tab_chat:
        render_chat_panel(client)

    with tab_docs:
        render_documents_panel(client)

    with tab_eval:
        render_eval_panel(client)


if __name__ == "__main__":
    main()