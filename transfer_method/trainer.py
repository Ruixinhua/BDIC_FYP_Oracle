import torch
import copy
import tools
import heapq


def backward(opt, loss):
    if train:
        # reset the gradients back to zero
        opt.zero_grad()
        # PyTorch accumulates gradients on subsequent backward passes
        loss.backward()
        opt.step()


def get_center_data(model, batch_data):
    centers = []
    for batch in batch_data:
        if model_type == "ae":
            batch = batch.view(-1, img_size * img_size)
        code, _ = tools.run_batch(model, batch.to(device), train=False, model_type=model_type)
        centers.append(torch.mean(code.cpu().data, dim=0).reshape((1, -1)))
    return centers


def predict():
    jia_centers, jin_centers = get_center_data(jia_model, jia_pred_data), get_center_data(jin_model, jin_full_data)
    correct_count, index_sum = 0, 0
    correct_char = {}
    for jia_center, jia_label in zip(jia_centers, jia_labels):

        distances_list = [tools.cal_dis_err(jia_center, jin_center, train=False, criterion=criterion) for jin_center in
                          jin_centers]
        top_n_char = map(distances_list.index, heapq.nsmallest(top_n, distances_list))
        predicted_chars = [jin_labels[i] for i in top_n_char]
        if jia_label in set(predicted_chars):
            correct_count += 1
            correct_index = predicted_chars.index(jia_label)
            index_sum += correct_index
            if correct_index not in correct_char:
                correct_char[correct_index] = []
            correct_char[correct_index].append(jia_label)
    accuracy = correct_count / len(jia_centers)
    # print("Current top %s accuracy: %.1f%%" % (top_n, accuracy * 100))
    return accuracy, index_sum, correct_char


def save_model(jia_path, jin_path):
    best_jia_model = copy.deepcopy(jia_model)
    best_jin_model = copy.deepcopy(jin_model)
    torch.save(best_jia_model.state_dict(), jia_path)
    torch.save(best_jin_model.state_dict(), jin_path)
    print("Save model to %s and %s" % (jia_path, jin_path))


def main():
    best_jia_model, best_jin_model = copy.deepcopy(jia_model), copy.deepcopy(jin_model)
    jia_batch_data, jin_batch_data, jia_jin_labels = tools.random_data(jia_batch_all, jin_batch_all, labels=labels_all,
                                                                       batch_level=batch_level, batch_size=batch_size)
    print("load data success!")
    best_acc, best_index_sum, correct_char = predict()
    print("Current top %s accuracy: %.1f%%, index sum: %s" % (top_n, best_acc * 100, best_index_sum))
    jia_loss_min, jin_loss_min = float("inf"), float("inf")

    print("Best model predict result:", correct_char)
    iter_no = 0
    for epoch in range(epochs):
        iter_no += 1
        # if no improvement, then stop
        # if epoch - min_epoch > 30:
        #     break
        loss, count = 0, 0
        jia_loss_recon, jia_count, jin_loss_recon, jin_count = 0, 0, 0, 0
        jia_loss_all, jin_loss_all = 0, 0
        ae_count, dis_count = 0, 0
        dis_err_all = 0
        for jia_batch, jin_batch, labels in zip(jia_batch_data, jin_batch_data, jia_jin_labels):
            # reshape mini-batch data to [N, 96*96] matrix
            # load it to the active device
            if model_type == "ae":
                jia_batch = jia_batch.view(-1, img_size * img_size).to(device)
                jin_batch = jin_batch.view(-1, img_size * img_size).to(device)
            else:
                jia_batch = jia_batch.to(device)
                jin_batch = jin_batch.to(device)

            if add_dis_cons:
                # fix jin model and train jia model
                jia_code, jia_loss = tools.run_batch(jia_model, jia_batch, train=train, model_type=model_type)
                jin_code, _ = tools.run_batch(jin_model, jin_batch, train=False, model_type=model_type)
                jia_loss += tools.cal_dis_err(jia_code, jin_code, labels=labels, train=train, criterion=criterion)
                backward(jia_optimizer, jia_loss)
                # fix jia model and train jin model
                jia_code, _ = tools.run_batch(jia_model, jia_batch, train=False, model_type=model_type)
                jin_code, jin_loss = tools.run_batch(jin_model, jin_batch, train=train, model_type=model_type)
                jin_loss += tools.cal_dis_err(jia_code, jin_code, labels=labels, train=train, criterion=criterion)
                backward(jin_optimizer, jin_loss)

                jia_loss_all += jia_loss.item()
                jin_loss_all += jin_loss.item()
                if train:
                    jia_code, jia_loss = tools.run_batch(jia_model, jia_batch, train=False, model_type=model_type)
                    jin_code, jin_loss = tools.run_batch(jin_model, jin_batch, train=False, model_type=model_type)
                # after update parameters, calculate distance error
                dis_err = tools.cal_dis_err(jia_code, jin_code, labels=labels, train=False, criterion=criterion)
            else:
                jin_code, jin_loss = tools.run_batch(jin_model, jin_batch, train=train, model_type=model_type)
                backward(jin_optimizer, jin_loss)
                jia_code, jia_loss = tools.run_batch(jia_model, jia_batch, train=train, model_type=model_type)
                backward(jia_optimizer, jia_loss)
                # after update parameters, calculate distance error
                dis_err = tools.cal_dis_err(jia_code, jin_code, labels=labels, train=False, criterion=criterion)
                jia_loss_all += jia_loss.item() + dis_err.item()
                jin_loss_all += jin_loss.item() + dis_err.item()
            if add_dis_cons and train:
                if not iter_no % 10:
                    accuracy, index_sum, correct_char = predict()
                    if accuracy >= best_acc:
                        if accuracy > best_acc or index_sum < best_index_sum:
                            best_acc = accuracy
                            best_index_sum = index_sum
                            print("Best model predict result:", correct_char)
                            save_model(jia_final_path, jin_final_path)
            # add the mini-batch training loss to epoch loss
            dis_err_all += dis_err.item()
            jia_loss_recon += jia_loss.item()
            jin_loss_recon += jin_loss.item()
            count += len(jia_batch)
        jia_batch_data, jin_batch_data, jia_jin_labels = tools.random_data(
            jia_batch_all, jin_batch_all, labels=labels_all, batch_level=batch_level, batch_size=batch_size)
        # compute the epoch training loss
        jia_loss_all /= count
        jia_loss_recon /= count
        jin_loss_all /= count
        jin_loss_recon /= count
        dis_err_all /= count
        if jia_loss_all < jia_loss_min and train:
            jia_loss_min = jia_loss_all
            torch.save(jia_model.state_dict(), jia_saved_path)
            print("saved model to %s" % jia_saved_path)
        if jin_loss_all < jin_loss_min and train:
            jin_loss_min = jin_loss_all
            torch.save(jin_model.state_dict(), jin_saved_path)
            print("saved model to %s" % jin_saved_path)
        # display the epoch training loss
        print("epoch:{}/{},jia recon loss:{:.4f},jin recon loss:{:.4f}".format(epoch + 1, epochs, jia_loss_recon, jin_loss_recon))
        print("epoch:{}/{},dis error:{:.4f},best acc:{:.1%}, index sum:{}".format(epoch + 1, epochs, dis_err_all, best_acc, best_index_sum))
    # print("Finished!jia final loss:{:.4f}, jin final loss:{:.4f}, dis loss:{:.4f}".format(jia_loss_final, jin_loss_final, dis_loss_final))


if __name__ == "__main__":
    train = True
    add_dis_cons = True  # second step training: add distance constrain
    batch_level = "all"  # the level of batch
    model_type = "ae"
    criterion = "mmd"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_char_num = 100
    batch_size = 512
    epochs = 100
    img_size = 96
    top_n = 10
    if model_type == "ae":
        # model_paths = ("model/jia_%s_final_%s_%s.pkl" % (model_type, batch_level, criterion),
        #                "model/jin_%s_final_%s_%s.pkl" % (model_type, batch_level, criterion))
        model_paths = ("model/jia_%s_base_adv.pkl" % model_type, "model/jin_%s_base_adv.pkl" % model_type)
        jia_saved_path, jin_saved_path = "model/jia_ae_base_adv.pkl", "model/jin_ae_base_adv.pkl"
        # jia_final_path, jin_final_path = "result/jia_ae_base_full.pkl", "result/jin_ae_base_full.pkl"
        jia_final_path = "model/jia_ae_final_%s_%s.pkl" % (batch_level, criterion)
        jin_final_path = "model/jin_ae_final_%s_%s.pkl" % (batch_level, criterion)
    else:
        model_paths = ("model/jia_%s_final_%s_%s_cons.pkl" % (model_type, batch_level, criterion),
                       "model/jin_%s_final_%s_%s_cons.pkl" % (model_type, batch_level, criterion))
        jia_saved_path, jin_saved_path = "model/jia_vae_base_adv.pkl", "model/jin_vae_base_adv.pkl"
        jia_final_path = "model/jia_vae_final_%s_%s.pkl" % (batch_level, criterion)
        jin_final_path = "model/jin_vae_final_%s_%s.pkl" % (batch_level, criterion)
    jia_model, jia_optimizer, jin_model, jin_optimizer, transform = tools.get_model_by_type(model_type, device, train, model_paths=model_paths)
    print("load model success!")
    # jia_batch_all, jin_batch_all, char_list = tools.get_paired_data(test_char_num=test_char_num, batch_level=batch_level, transform=transform)
    jia_batch_all, jin_batch_all, labels_all, char_list = tools.get_paired_data(test_char_num=test_char_num, batch_level=batch_level, transform=transform, labeled=True)
    jia_pred_data, jia_labels = tools.get_data_by_type("jia", set(char_list[test_char_num:]), transform=transform)
    jin_full_data, jin_labels = tools.get_data_by_type("jin", set(char_list), transform=transform)
    main()
