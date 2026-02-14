"""
===============================================================================
DATA WARMER - Big Tech Hot/Cold Pattern (Snowflake/Databricks)
===============================================================================

Purpose: Load data from B2 (cold) to Neon (hot) when user returns after 24h.

Big Tech Pattern:
- Snowflake: Auto-resume suspended warehouses
- Databricks: Delta Lake automatic file compaction
- Netflix: Re-fetch content when cache expires

Flow:
1. User asks "analyze sales data"
2. Agent calls ensure_data_loaded(user_id, file_name)
3. Check if data exists in Neon (HOT)
4. If not, download from B2 (COLD) → load to Neon
5. Set fresh 24h TTL
6. Agent queries Neon

Safety:
- Multi-tenant: Always filter by user_id
- Idempotent: Safe to call multiple times
- Transactional: All or nothing insert
- Cleanup: Temp files always deleted

Created: Feb 2026
Contest: NVIDIA GTC 2026 Golden Ticket
===============================================================================
"""

import os
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from tools.storage.b2_client import get_b2_client, B2Response

logger = logging.getLogger(__name__)


# =============================================================================
# RESPONSE DATACLASS
# =============================================================================

@dataclass
class WarmUpResult:
    """Result of data warm-up operation."""
    success: bool = False
    source: str = ""  # "hot" (already in Neon) or "cold" (fetched from B2)
    rows_loaded: int = 0
    file_name: str = ""
    error: Optional[str] = None
    operation_time_ms: int = 0


# =============================================================================
# CONFIGURATION
# =============================================================================

# TTL for warmed-up data (same as fresh upload)
DATA_TTL_HOURS = 24

# Temp directory for B2 downloads
TEMP_DIR = "/tmp/lubot_warmup"

# Batch size for inserts (Big Tech: avoid memory issues with large files)
BATCH_SIZE = 1000


# =============================================================================
# MAIN FUNCTION: ensure_data_loaded
# =============================================================================

def ensure_data_loaded(
    user_id: str,
    file_name: str,
    db: Session
) -> WarmUpResult:
    """
    Big Tech Hot/Cold Pattern: Ensure data is loaded in Neon before query.

    This is the SAME behavior as fresh upload, just a different route:
    - Route 1 (Upload): User's computer → Neon
    - Route 2 (Warm Up): B2 cold storage → Neon

    Args:
        user_id: User's unique ID (multi-tenant isolation)
        file_name: Name of the file to load
        db: SQLAlchemy database session

    Returns:
        WarmUpResult with success status and metadata

    Example:
        result = ensure_data_loaded("abc123", "sales.csv", db)
        if result.success:
            # Data is now in Neon, ready to query
            df = query_user_data(user_id, file_name)
    """
    import time
    start_time = time.time()

    try:
        # =====================================================================
        # STEP 1: Check if data is HOT (already in Neon)
        # =====================================================================
        logger.info(f"🔍 Checking HOT storage for {file_name} (user {user_id[:8]}...)")

        hot_count = db.execute(text("""
            SELECT COUNT(*) FROM user_uploaded_rows
            WHERE user_id = :user_id
            AND upload_id IN (
                SELECT id::text FROM user_uploads
                WHERE user_id = :user_id AND filename = :file_name
            )
        """), {"user_id": user_id, "file_name": file_name}).scalar() or 0

        if hot_count > 0:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"✅ Data is HOT: {hot_count} rows in Neon ({elapsed_ms}ms)")
            return WarmUpResult(
                success=True,
                source="hot",
                rows_loaded=hot_count,
                file_name=file_name,
                operation_time_ms=elapsed_ms
            )

        # =====================================================================
        # STEP 2: Check METADATA (user_file_schemas)
        # =====================================================================
        logger.info(f"❄️ Data is COLD, checking metadata...")

        metadata = db.execute(text("""
            SELECT storage_path, file_type, columns, column_types, row_count
            FROM user_file_schemas
            WHERE user_id = :user_id AND file_name = :file_name
        """), {"user_id": user_id, "file_name": file_name}).fetchone()

        if not metadata:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"❌ File not found in metadata: {file_name}")
            return WarmUpResult(
                success=False,
                source="none",
                file_name=file_name,
                error=f"File '{file_name}' not found. Please upload it first.",
                operation_time_ms=elapsed_ms
            )

        storage_path = metadata[0]
        file_type = metadata[1]
        expected_rows = metadata[4] or 0

        if not storage_path:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"❌ No storage path for file: {file_name}")
            return WarmUpResult(
                success=False,
                source="none",
                file_name=file_name,
                error=f"File '{file_name}' has no cold storage backup.",
                operation_time_ms=elapsed_ms
            )

        # =====================================================================
        # STEP 3: Download from B2 (COLD)
        # =====================================================================
        logger.info(f"📥 Downloading from B2: {storage_path}")

        # Create temp directory
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

        # Temp file path (includes user_id for isolation)
        safe_filename = Path(file_name).name.replace("..", "").replace("/", "_")
        temp_path = f"{TEMP_DIR}/{user_id}_{safe_filename}"

        # Download from B2
        b2_client = get_b2_client()
        download_result = b2_client.download_user_file(
            user_id=user_id,
            filename=file_name,
            local_path=temp_path
        )

        if not download_result.success:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"❌ B2 download failed: {download_result.error}")
            return WarmUpResult(
                success=False,
                source="cold",
                file_name=file_name,
                error=f"Could not retrieve file from cold storage: {download_result.error}",
                operation_time_ms=elapsed_ms
            )

        # =====================================================================
        # STEP 4: Parse file (CSV or Excel)
        # =====================================================================
        logger.info(f"📊 Parsing file: {file_type}")

        try:
            if file_type in ['xlsx', 'xls']:
                df = pd.read_excel(temp_path)
            else:
                # Try UTF-8 first, then fallback to latin-1
                try:
                    df = pd.read_csv(temp_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(temp_path, encoding='latin-1')

            logger.info(f"📊 Parsed {len(df)} rows, {len(df.columns)} columns")

        except Exception as e:
            # Cleanup temp file
            _cleanup_temp_file(temp_path)
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"❌ Parse failed: {e}")
            return WarmUpResult(
                success=False,
                source="cold",
                file_name=file_name,
                error=f"Could not parse file: {str(e)}",
                operation_time_ms=elapsed_ms
            )

        # =====================================================================
        # STEP 5: Load to Neon (HOT) with fresh 24h TTL
        # =====================================================================
        logger.info(f"📤 Loading {len(df)} rows to Neon (24h TTL)...")

        try:
            rows_loaded = _load_to_neon(
                df=df,
                user_id=user_id,
                file_name=file_name,
                file_type=file_type,
                db=db
            )

            logger.info(f"✅ Loaded {rows_loaded} rows to Neon")

        except Exception as e:
            # Cleanup temp file
            _cleanup_temp_file(temp_path)
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error(f"❌ Load to Neon failed: {e}")
            return WarmUpResult(
                success=False,
                source="cold",
                file_name=file_name,
                error=f"Could not load data to Neon: {str(e)}",
                operation_time_ms=elapsed_ms
            )

        # =====================================================================
        # STEP 6: Cleanup temp file
        # =====================================================================
        _cleanup_temp_file(temp_path)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"✅ Warm-up complete: {rows_loaded} rows in {elapsed_ms}ms")

        return WarmUpResult(
            success=True,
            source="cold",
            rows_loaded=rows_loaded,
            file_name=file_name,
            operation_time_ms=elapsed_ms
        )

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Warm-up failed: {e}")
        import traceback
        traceback.print_exc()
        return WarmUpResult(
            success=False,
            source="error",
            file_name=file_name,
            error=str(e),
            operation_time_ms=elapsed_ms
        )


# =============================================================================
# HELPER: Load DataFrame to Neon
# =============================================================================

def _load_to_neon(
    df: pd.DataFrame,
    user_id: str,
    file_name: str,
    file_type: str,
    db: Session
) -> int:
    """
    Load DataFrame to user_uploaded_rows with fresh 24h TTL.

    Big Tech Pattern: Same behavior as fresh upload (Route 1).
    - Creates new upload record in user_uploads
    - Inserts rows to user_uploaded_rows
    - Sets expires_at to NOW() + 24 hours

    Args:
        df: Parsed DataFrame
        user_id: User ID
        file_name: File name
        file_type: File type (csv, xlsx, xls)
        db: Database session

    Returns:
        Number of rows loaded
    """
    import json
    import numpy as np

    # Generate new upload_id for this warm-up
    upload_id = str(uuid.uuid4())

    # =========================================================================
    # Step 1a: Extract column metadata from DataFrame
    # Big Tech Pattern: Same as fresh upload - detect column types
    # =========================================================================
    all_columns = list(df.columns)
    numeric_columns = []
    dimension_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            dimension_columns.append(col)

    logger.info(f"📊 Column metadata: {len(all_columns)} total, {len(numeric_columns)} numeric, {len(dimension_columns)} dimensions")

    # =========================================================================
    # Step 1b: Delete existing rows for this file (idempotent)
    # Big Tech: "Last load wins" - prevent duplicates
    # =========================================================================
    db.execute(text("""
        DELETE FROM user_uploaded_rows
        WHERE user_id = :user_id
        AND upload_id IN (
            SELECT id::text FROM user_uploads
            WHERE user_id = :user_id AND filename = :file_name
        )
    """), {"user_id": user_id, "file_name": file_name})

    # Also delete old upload records for this file
    db.execute(text("""
        DELETE FROM user_uploads
        WHERE user_id = :user_id AND filename = :file_name
    """), {"user_id": user_id, "file_name": file_name})

    # =========================================================================
    # Step 2: Create upload record with 24h TTL + column metadata
    # =========================================================================
    db.execute(text("""
        INSERT INTO user_uploads (
            id, user_id, filename, file_path, file_type,
            rows_total, rows_imported, status,
            storage_mode, raw_storage_type,
            all_columns, numeric_columns, dimension_columns,
            has_date_column,
            created_at, expires_at
        ) VALUES (
            :id, :user_id, :filename, :file_path, :file_type,
            :rows_total, 0, 'warming_up',
            'universal', 'b2',
            :all_columns, :numeric_columns, :dimension_columns,
            false,
            NOW(), NOW() + INTERVAL '24 hours'
        )
        ON CONFLICT (id) DO NOTHING
    """), {
        "id": upload_id,
        "user_id": user_id,
        "filename": file_name,
        "file_path": f"b2://{user_id}/{file_name}",
        "file_type": file_type,
        "rows_total": len(df),
        "all_columns": json.dumps(all_columns),
        "numeric_columns": json.dumps(numeric_columns),
        "dimension_columns": json.dumps(dimension_columns)
    })

    # =========================================================================
    # Step 3: Insert rows in batches
    # Big Tech: Batch inserts to avoid memory issues with large files
    # =========================================================================
    rows_imported = 0
    total_rows = len(df)

    for batch_start in range(0, total_rows, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        batch_df = df.iloc[batch_start:batch_end]

        values_list = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            row_index = batch_start + idx
            row_id = f"{upload_id}_{row_index}"

            # Convert row to JSON (handle NaN, dates, etc.)
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = val

            row_data = json.dumps(row_dict)
            # Escape single quotes for SQL
            row_data_escaped = row_data.replace("'", "''")

            values_list.append(
                f"('{row_id}', '{user_id}', '{upload_id}', "
                f"{row_index}, '{row_data_escaped}', NOW(), NOW())"
            )

        if values_list:
            values_str = ',\n'.join(values_list)
            db.execute(text(f"""
                INSERT INTO user_uploaded_rows
                (row_id, user_id, upload_id, row_index, row_data, created_at, updated_at)
                VALUES {values_str}
            """))
            rows_imported += len(values_list)

        logger.info(f"   Loaded batch: {rows_imported}/{total_rows} rows")

    # =========================================================================
    # Step 4: Update upload record status
    # =========================================================================
    db.execute(text("""
        UPDATE user_uploads
        SET status = 'completed',
            rows_imported = :rows_imported
        WHERE id = :id
    """), {"id": upload_id, "rows_imported": rows_imported})

    db.commit()

    return rows_imported


# =============================================================================
# HELPER: Cleanup temp file
# =============================================================================

def _cleanup_temp_file(temp_path: str) -> None:
    """
    Delete temp file after use.

    Big Tech Pattern: Always cleanup temp files to prevent disk fill.
    """
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"🧹 Cleaned up temp file: {temp_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not cleanup temp file: {e}")


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DATA WARMER - Test")
    print("=" * 60)
    print("This module is meant to be imported, not run directly.")
    print("Usage:")
    print("  from tools.storage.data_warmer import ensure_data_loaded")
    print("  result = ensure_data_loaded(user_id, file_name, db)")
    print("=" * 60)
