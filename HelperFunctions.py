import torch

torch.backends.cudnn.enabled = False
import torch.utils.data as torch_data
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pytorch_nsynth.nsynth import NSynth
import itertools
import os
import numpy as np


def create_directory(directory_name):
    """
    If given directory does not exists, this function will create a directory with given name.
    :param directory_name: name of the required directory
    :return: None
    """
    directory = "./" + directory_name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_waveforms(data, name, title, directory_name):
    """
    Plots a 2d graph for the given data. This function is being used to plot waveforms.
    :param data: 1-dimensional data for plotting
    :param name: name of the plot's file
    :param title: title of the plot
    :param directory_name: name of the directory providing location information to save plot
    :return: None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(data)
    plt.title(title)
    plt.savefig('./' + directory_name + '/' + name)
    pass


def get_loader(path, batch_size, device, directory_name):
    """
    Helper function to load audio samples and rescale. This function also subsamples data by using initial 16000
    intensities.
    :param path: path to the audio data
    :param batch_size: how many samples per batch to load
    :param device: specifies torch device being used - CPU or CUDA
    :param directory_name: specifies location to save sample waveform plot
    :return: numpy array of audio samples, dataloader on the dataset(numpy array)

    Reference for the function: https://github.com/kwon-young/pytorch-nsynth
    """
    if str(device) == "cpu":
        path = "/home/amit/PycharmProjects/ML_Project_1/nsynth-test"
    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    toSelectCols = transforms.Lambda(lambda x: x[0:16000])
    toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max + 1)
    # use instrument_family and instrument_source as classification targets
    dataset = NSynth(
        path,
        transform=transforms.Compose([toSelectCols, toFloat]),
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])

    print(path, "Length: ", len(dataset))
    plot_waveforms(dataset[0][0], "1-D_audio_waveform.png", "1-D audio waveform", directory_name)
    return dataset, torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def plot(training_losses, validation_losses, epochs, directory_name):
    """
    This function plots learning curve for training and validation.
    :param training_losses: array of training losses
    :param validation_losses: array of validation losses
    :param epochs: number of epochs specifies x-axis for the plot
    :param directory_name: specifies location to save the plot
    :return: None
    """
    plt.figure(figsize=(20, 10))

    x = np.linspace(1, epochs, epochs)
    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)

    plt.title("Learning curve over Epochs")

    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")

    plt.plot(x, training_losses, color='purple', marker=".", label='Training loss')
    plt.plot(x, validation_losses, color='orange', marker=".", label='Validation loss')
    plt.legend()
    plt.savefig('./' + directory_name + '/Learning_curves-' + str(epochs) + '.png')
    pass


def plot_confusion_matrix(y_targeted, y_predicted, instrument_classes, directory_name):
    """
    This function plots heat-map for the confusion matrix generated using y_targeted and y_predicted data
    :param y_targeted: True labels
    :param y_predicted: Predicted labels
    :param instrument_classes: Ticks for x-axis and y-axis
    :param directory_name: specifies location to save the plot
    :return: None

    Reference for the function: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(y_targeted, y_predicted)
    np.set_printoptions(precision=2)
    cm = cm / cm.astype(np.float).sum(axis=1)
    print("Normalized confusion matrix:\n", cm)

    fig = plt.figure(figsize=(10, 7))
    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(instrument_classes))
    plt.xticks(tick_marks, instrument_classes, rotation=45)
    plt.yticks(tick_marks, instrument_classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.title("Confusion matrix")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    fig.savefig('./' + directory_name + '/confusion_matrix')
    pass


def correct_class_probability_waveforms_plot(classes_max_tensor, classes_min_tensor, instrument_classes,
                                             directory_name):
    """
    Plots waveforms given in classes_max_tensor and classes_min_tensor
    :param classes_max_tensor: tensors of waveforms for each class where the correct class probability is very high
    :param classes_min_tensor: tensors of waveforms for each class where the correct class probability is very low
    :param instrument_classes: list of instruments used for title of plots
    :param directory_name: specifies location to save the plot
    :return: None
    """
    for i in range(len(classes_max_tensor)):
        if classes_max_tensor[i] is None:
            continue
        plot_waveforms(classes_max_tensor[i].data.cpu().numpy(), "class_" + instrument_classes[i] +
                       "_max_probability_waveform", "class_" + instrument_classes[i] +
                       "_max_probability_waveform", directory_name)
    for i in range(len(classes_min_tensor)):
        if classes_min_tensor[i] is None:
            continue
        plot_waveforms(classes_min_tensor[i].data.cpu().numpy(), "class_" + instrument_classes[i] +
                       "_min_probability_waveform", "class_" + instrument_classes[i] +
                       "_min_probability_waveform", directory_name)
    pass


def plot_waveforms_for_samples_near_decision_boundary(sample_waveforms, instrument_classes, directory_name):
    """
    For each class, this function plots waveforms for samples near the decision boundary, where the probability for the
    correct class is slightly higher/lower than the other classes.
    :param sample_waveforms: list of waveforms (type: Tensor)
    :param instrument_classes: list of instruments used for title of plots
    :param directory_name: specifies location to save the plot
    :return: None
    """
    sample_waveforms.sort(key=lambda tup: abs(tup[2]))
    waveform_plotted_status = [False] * len(instrument_classes)
    for label, nearest_class, difference, waveform in sample_waveforms:
        if not waveform_plotted_status[label]:
            plot_waveforms(waveform, "Near_decision_boundary_sample_for_class_" + instrument_classes[label],
                           "Label class: " + instrument_classes[label] + " Nearby class: "
                           + instrument_classes[nearest_class] + " Difference: " + str(difference), directory_name)
            waveform_plotted_status[label] = True
    pass


def train(net, device, batch_size, epochs, criterion, optimizer, directory_name):
    """
    This function loads training and validation loader to perform train network. For each epoch, the network is trained
    on the input data and later validated. At the end, we plot learning curve based on training and validation losses.
    :param net: network model's object that will be trained on training set and validated on validation set
    :param device: specifies torch device being used - CPU or CUDA
    :param batch_size: samples per batch
    :param epochs: number of training cycles
    :param criterion: Specifies function to calculate loss
    :param optimizer: Specifies an optimizer object that will hold the current state and will update the parameters
    based on the computed gradients.
    :param directory_name: specifies location to save the plot
    :return: None
    """
    create_directory(directory_name)

    # Loaders
    train_data, train_loader = get_loader("/local/sandbox/nsynth/nsynth-train", batch_size, device, directory_name)
    validation_data, validation_loader = get_loader("/local/sandbox/nsynth/nsynth-valid", batch_size, device,
                                                    directory_name)

    print("Loading data complete.")
    print("Training begins")
    # Train the network
    training_losses = []
    validation_losses = []

    for epoch in range(1, epochs + 1):
        print("Epoch:", epoch)
        train_loss = 0.0
        correct = 0
        total = 0
        for samples, instrument_family_target, instrument_source_target, targets in train_loader:
            # Get the inputs
            inputs, labels = samples.to(device), instrument_family_target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)

            # statistics
            total += inputs.size(0)
            correct += (preds == labels).sum().item()
            train_loss += loss.item()

        epoch_acc = (float(correct) * 100) / total
        epoch_loss = float(train_loss) / len(train_data)
        training_losses.append(epoch_loss)
        print("Train Loss:", epoch_loss, "Accuracy:", epoch_acc)

        correct = 0
        total = 0
        validation_loss = 0.0

        with torch.no_grad():
            for samples, instrument_family_target, instrument_source_target, targets in validation_loader:
                # Get the inputs
                inputs, labels = samples.to(device), instrument_family_target.to(device)

                val_outputs = net(inputs)
                loss = criterion(val_outputs, labels)

                _, preds = torch.max(val_outputs, 1)

                total += inputs.size(0)
                correct += (preds == labels).sum().item()
                validation_loss += loss.item()

        epoch_acc = (float(correct) * 100) / total
        epoch_loss = float(validation_loss) / len(validation_data)
        validation_losses.append(epoch_loss)
        print("Validation Loss:", epoch_loss, "Accuracy:", epoch_acc)
        torch.save(net.state_dict(), "./" + directory_name.replace("images", "states") + "_" + str(epoch))
        plot(training_losses, validation_losses, epoch, directory_name)

    print('Finished training')
    pass


def test(net, device, batch_size, directory_name):
    """
    This function loads testing set to test the given network model's object and generate confusion matrix and graphs
    with min and max probability for each class and waveforms for samples near the decision boundary.
    :param net: network model's object that will be tested on test set
    :param device: specifies torch device being used - CPU or CUDA
    :param batch_size: samples per batch
    :param directory_name: specifies location to save the plot
    :return: None
    """
    create_directory(directory_name)

    test_data, test_loader = get_loader("/local/sandbox/nsynth/nsynth-test", batch_size, device, directory_name)

    print("Loading data complete.")
    print("Testing begins")

    correct = 0
    total = 0
    classes = set()
    with torch.no_grad():
        for samples, instrument_family_target, instrument_source_target, targets in test_loader:
            for label in instrument_family_target:
                classes.add(label.item())
            inputs, labels = samples.to(device), instrument_family_target.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the test files: %d %%\n' % (
            100 * correct / total))

    instrument_classes = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'vocal']
    classes = list(classes)
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    y_targeted = []
    y_predicted = []
    classes_max = [0] * len(classes)
    classes_min = [1] * len(classes)
    classes_max_tensor = [None] * len(classes)
    classes_min_tensor = [None] * len(classes)
    # waveforms with probabilities near decision boundaries
    sample_waveforms = []

    with torch.no_grad():
        for samples, instrument_family_target, instrument_source_target, targets in test_loader:
            inputs, labels = samples.to(device), instrument_family_target.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            # confusion matrix
            y_targeted.extend(labels)
            y_predicted.extend(predicted)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                # graphs for max-min class probability
                correct_class_probability = outputs[i][label].item()
                if correct_class_probability > classes_max[label.item()]:
                    classes_max[label.item()] = correct_class_probability
                    classes_max_tensor[label.item()] = inputs[i]
                if correct_class_probability < classes_min[label.item()]:
                    classes_min[label.item()] = correct_class_probability
                    classes_min_tensor[label.item()] = inputs[i]

                # graphs for probabilities near decision boundaries
                probabilities = outputs[i].data.cpu().numpy()
                target_probability = probabilities[label.item()]
                thresold = 0.3
                for index in range(len(probabilities)):
                    difference = abs(target_probability - probabilities[index])
                    if index != label.item() and abs(difference) < thresold:
                        sample_waveforms.append((label.item(), index, difference,
                                                 inputs[i].data.cpu().numpy()))
                        break

    classes_accuracy = []
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        classes_accuracy.append(100 * class_correct[i] / class_total[i])

    plot_confusion_matrix(y_targeted, y_predicted, instrument_classes, directory_name)
    correct_class_probability_waveforms_plot(classes_max_tensor, classes_min_tensor, instrument_classes, directory_name)
    plot_waveforms_for_samples_near_decision_boundary(sample_waveforms, instrument_classes, directory_name)
    print("Finished testing")
    pass
