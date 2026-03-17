import importlib

from src.utils.logging import get_logger

logger = get_logger("Eval runner scaffold")

def main(eval_name, eval_file, args_eval, resume_preempt=False):
    logger.info(f"Running evaluation: {eval_name}")
    import_path = f"evals.{eval_name}.{eval_file}"
    return importlib.import_module(import_path).main(args_eval=args_eval, resume_preempt=resume_preempt)
