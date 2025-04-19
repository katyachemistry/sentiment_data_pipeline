from dagster import asset, AssetExecutionContext, op, job, repository, In, Out, String
import pandas as pd
from scripts.annotation import *

@asset(group_name="autom_annotation")
def auto_annotation(context: AssetExecutionContext, anomaly_cleaning: pd.DataFrame) -> pd.DataFrame:
    """Auto-annotation, complexity analysis and selection of data for manual annotation (uncertainty threshold 0.6)"""
    try:
        clean_data = anomaly_cleaning
        annotated_data = auto_annotate(clean_data)
        confident, non_confident = analyze_complexity(annotated_data)
        context.log.info(f"Auto-annotation, complexity analysis and selection of data for manual annotation complete")
        return confident
    except Exception as e:
        context.log.info(f"❌ Error during annotation: {str(e)}")
        raise