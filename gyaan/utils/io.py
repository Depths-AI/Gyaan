import polars as pl
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
import asyncio
from typing import Optional, List

async def create_table(
    table_path: str,
    data: pl.DataFrame,
    mode: Optional[str]="ignore",
    num_retries: Optional[int]=3):

    assert table_path.startswith("file://"), "Table path must be a file URI"
    
    for attempt in range(num_retries):
        try:
            data.write_delta(
                table_path, 
                mode=mode)
            break
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt+1)*0.1)
    
async def read_table(
    table_path: str):

    assert table_path.startswith("file://"), "Table path must be a file URI"
    
    try:
        dt = DeltaTable(table_path)
        pyarrow_table = dt.to_pyarrow_table()
        df = pl.from_arrow(pyarrow_table)
        return df
    except Exception as e:
        print(f"Error reading with deltalake library directly: {e}")
    
async def insert_table(
    table_path:str,
    insertion_df: pl.DataFrame,
    num_retries: Optional[int]=3):
    
    assert table_path.startswith("file://"), "Table path must be a file URI"
    assert insertion_df.height>0, "Data to be inserted should be non-empty"

    for attempt in range(num_retries):
        try:
            insertion_df.write_delta(
                table_path, 
                mode="append")
            break
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt+1)*0.1)

async def update_table(
    table_path:str,
    update_df: pl.DataFrame,
    id_column: str="id",
    num_retries: Optional[int]=3):

    assert table_path.startswith("file://"), "Table path must be a file URI"
    assert update_df.height>0, "Data to be updated should be non-empty"
    
    predicate=f"source.{id_column}=target.{id_column}"
    update_set = {col: f"source.{col}" for col in update_df.columns}

    for attempt in range(num_retries):
        try:
            update_df.write_delta(
                table_path,
                mode="merge",
                delta_merge_options={
                    "predicate": predicate,
                    "source_alias": "source",
                    "target_alias": "target"
                }).when_matched_update(updates=update_set).execute()
            break
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt+1)*0.1)

async def soft_delete(
    table_path:str,
    delete_df: pl.DataFrame,
    predicate:str="source.id=target.id",
    num_retries: Optional[int]=3):
    
    assert table_path.startswith("file://"), "Table path must be a file URI"
    assert predicate is not None, "Predicate must be provided"
    
    for attempt in range(num_retries):
        try:
            delete_df.write_delta(
                table_path,
                mode="merge",
                delta_merge_options={
                    "predicate": predicate,
                    "source_alias": "source",
                    "target_alias": "target"
                }).when_matched_update(updates={"deleted":"source.deleted"}).execute()
            break
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt+1)*0.1)

async def optimize(
    table_path:str,
    z_order_index: Optional[List[str]]=None,
    retention_hours: Optional[int]=24,
    dry_delete: Optional[bool]=True,
    num_retries: Optional[int]=3):
    
    assert table_path.startswith("file://"), "Table path must be a file URI"
    
    for attempt in range(num_retries):
        try:
            dt = DeltaTable(table_path)
            if z_order_index is not None:
                stats=dt.optimize.z_order(z_order_index)
            else:
                stats=dt.optimize.compact()
            dt.vacuum(retention_hours=retention_hours, dry_run=dry_delete, enforce_retention_duration=False)
            return stats
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt+1)*0.1)