import torch
def save_checkpoint(
    model,
    optimizer,
    scheduler,
    total_steps_done,
    epochs_done,
    config,
    output_path,
):
    """Save checkpoint"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "total_steps_done": total_steps_done,
        "epochs_done": epochs_done,
        "config": config,
    }

    checkpoint_path = f"{output_path}/checkpoint_step_{total_steps_done}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    total_steps_done = checkpoint["total_steps_done"]
    epochs_done = checkpoint["epochs_done"]
    config = checkpoint['config']

    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(
        f"📊 Restored state: Step {total_steps_done}, Epoch {epochs_done}"
    )
    return model, optimizer, scheduler, total_steps_done, epochs_done,config

