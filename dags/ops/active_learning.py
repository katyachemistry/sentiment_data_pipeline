from dagster import op, In, Out, String
from scripts.active_learning import load_data, run_active_learning


@op(ins={"batch_num": In(int), "new_labels_path": In(String), "n_queries": In(int), "save_model": In(bool)}, out=Out(bool))
def active_learning_round(context, batch_num: int, new_labels_path: str, n_queries: int, save_model:bool) -> bool:
    try:
        if batch_num == 0:
            context.log.info("🔁 First launch of active learning.")
            data = load_data(first_launch=True)
        else:
            context.log.info(f"🔁 Loading data with new labels from: {new_labels_path}")
            data = load_data(first_launch=False, new_labels_path=new_labels_path)

        run_active_learning(data, batch_num=batch_num, n_queries=n_queries, save_model=save_model)
        context.log.info(f"✅ Active learning round {batch_num} complete.")
        return True
    except Exception as e:
        context.log.error(f"❌ Failed active learning round {batch_num}: {str(e)}")
        raise