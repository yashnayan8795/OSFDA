"""
Manual Test Data Upload — shared sidebar component.

Call render_manual_test_sidebar() from every page (app.py + all pages/).
Lets users pick a problem, see the required data format, upload a CSV/Parquet,
and navigate to the appropriate problem page with the data loaded.

Session state keys:
    mt_uploaded_dfs:       dict[problem_id, pd.DataFrame]
    mt_pending_navigation: str | None  — consumed once, then cleared
    mt_last_upload_id:     str | None  — prevents re-triggering on rerun
"""
from __future__ import annotations

import streamlit as st

from .schemas import SCHEMAS, UNSUPPORTED_PROBLEMS, get_schema
from .validation import read_upload, validate, MAX_ROWS

PAGE_PATHS: dict[str, str] = {
    "severity":  "pages/1_Problem_A_Severity.py",
    "preflight": "pages/3_Problem_C_Preflight.py",
}


def _init_state() -> None:
    if "mt_uploaded_dfs" not in st.session_state:
        st.session_state.mt_uploaded_dfs = {}
    if "mt_pending_navigation" not in st.session_state:
        st.session_state.mt_pending_navigation = None
    if "mt_last_upload_id" not in st.session_state:
        st.session_state.mt_last_upload_id = None


def get_uploaded_df(problem_id: str):
    """Return the uploaded DataFrame for this problem, or None."""
    _init_state()
    return st.session_state.mt_uploaded_dfs.get(problem_id)


def clear_uploaded_df(problem_id: str) -> None:
    """Clear uploaded data for a problem (e.g. from a reset button)."""
    _init_state()
    st.session_state.mt_uploaded_dfs.pop(problem_id, None)


def _consume_pending_navigation() -> None:
    target = st.session_state.mt_pending_navigation
    if target:
        st.session_state.mt_pending_navigation = None
        try:
            st.switch_page(target)
        except Exception:
            st.sidebar.warning(
                f"Could not navigate to `{target}`. Open the page manually."
            )


def _render_format_help(problem_id: str) -> None:
    schema = get_schema(problem_id)
    if schema is None:
        st.info(
            f"Manual upload is not yet supported for "
            f"**{UNSUPPORTED_PROBLEMS.get(problem_id, problem_id)}**. "
            f"Pick Problem A or Problem C."
        )
        return

    st.caption(schema.notes)

    with st.expander("Required columns (strict)", expanded=False):
        for col in schema.required:
            st.markdown(f"- `{col}`")

    if schema.optional:
        with st.expander("Optional columns (filled with NaN if missing)", expanded=False):
            for col in schema.optional:
                st.markdown(f"- `{col}`")

    if schema.target:
        st.caption(
            f"Include a `{schema.target}` column for evaluation metrics. "
            f"Without it, only predictions will be shown."
        )


def render_manual_test_sidebar() -> None:
    """Render the Manual Test section in the sidebar. Call from every page."""
    _init_state()
    _consume_pending_navigation()

    with st.sidebar:
        st.divider()
        with st.expander("🧪 Manual Test", expanded=False):
            st.caption(
                f"Upload your own dataset to run the supported models. "
                f"Max **{MAX_ROWS:,} rows**, CSV or Parquet."
            )

            options = []
            for pid, schema in SCHEMAS.items():
                options.append((pid, schema.display_name))
            for pid, name in UNSUPPORTED_PROBLEMS.items():
                options.append((pid, f"{name} (not yet supported)"))

            labels = [name for _, name in options]
            ids = [pid for pid, _ in options]

            choice_idx = st.selectbox(
                "Problem",
                range(len(options)),
                format_func=lambda i: labels[i],
                key="mt_problem_select",
            )
            problem_id = ids[choice_idx]

            _render_format_help(problem_id)

            if problem_id not in SCHEMAS:
                return

            uploaded = st.file_uploader(
                "Upload CSV or Parquet",
                type=["csv", "parquet"],
                key=f"mt_upload_{problem_id}",
            )

            existing = st.session_state.mt_uploaded_dfs.get(problem_id)
            if existing is not None and uploaded is None:
                st.success(
                    f"Active dataset: {len(existing):,} rows × "
                    f"{len(existing.columns)} columns"
                )
                if st.button("Clear uploaded data", key=f"mt_clear_{problem_id}"):
                    clear_uploaded_df(problem_id)
                    st.session_state.mt_last_upload_id = None
                    st.rerun()
                return

            if uploaded is None:
                return

            file_id = getattr(uploaded, "file_id", None) or uploaded.name
            already_consumed = (
                st.session_state.mt_last_upload_id == file_id
                and existing is not None
            )
            if already_consumed:
                return

            df, read_err = read_upload(uploaded)
            if read_err:
                st.error(read_err)
                return

            schema = get_schema(problem_id)
            result = validate(df, schema)

            if not result.ok:
                for msg in result.errors:
                    st.error(msg)
                return

            for w in result.warnings:
                st.warning(w)

            st.session_state.mt_uploaded_dfs[problem_id] = result.df
            st.session_state.mt_last_upload_id = file_id
            st.success(
                f"Loaded {len(result.df):,} rows. Switching to "
                f"{schema.display_name}…"
            )

            target = PAGE_PATHS.get(problem_id)
            if target:
                st.session_state.mt_pending_navigation = target
                st.rerun()
