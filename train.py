def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, (data,label) in enumerate(epoch_iterator_val):
            val_inputs, val_labels = data.cuda(),label.cuda()
            val_labels.unsqueeze_(1)
            val_outputs = sliding_window_inference(val_inputs, (96, 128, 128), 3, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            hand_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        hand_dice_val = hand_metric.aggregate().item()
        dice_metric.reset()
        hand_metric.reset()
    return mean_dice_val,hand_dice_val


def train(global_step, train_loader, dice_val_best,hand_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, (data,label) in enumerate(epoch_iterator):
        step += 1
        x, y = data.cuda(),label.cuda()
        y.unsqueeze_(1)
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val,hand_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            metric_values2.append(hand_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join("/content/drive/MyDrive/Colab Notebooks/Pro/experiments", "test2.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            if hand_val < hand_val_best:
              hand_val_best = hand_val
        global_step += 1
    return global_step, dice_val_best, global_step_best
max_iterations = 5500
eval_num = 500
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hand_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)

global_step = 0

dice_val_best = 0.0
hand_val_best = 100.0
global_step_best = 0
epoch_loss_values = []


metric_values = []
metric_values2 = []

while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best,hand_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join("/content/drive/MyDrive/Colab Notebooks/Pro/experiments", "test2.pth")))