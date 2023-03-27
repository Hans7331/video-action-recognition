import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as F

# for activitynet 
class ConfusionMatrix:
    
    # for creating dataframe of confusion matrix
    def create_cm(num_classes):
        class_list = [x for x in range(num_classes)]
        cm = pd.DataFrame(0, columns = class_list, index = class_list)
        return cm
    # for updating values in it
    def update_cm(cm, labels, prediction):
        for lb, pr in zip(labels, prediction):
            cm[pr][lb] += 1 
        return cm
    
    # for convering it to percentage class-wise
    def percentage_cm(idx_to_class, class_length, confusion_matrix):
        if len(class_length) == 0:
            return "-"
        return confusion_matrix.div(confusion_matrix.sum(axis = 1), axis = 0)

    
# for UCF101
def plot_confuse_matrix(matrix, classes,
                        s_path = "confuse.png",
                        normalize=True,
                        title=None,
                        cmap=plt.cm.Blues
                        ):
    """
    :param matrix:
    :param classes:
    :param s_path:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()

    # We change the fontsize of minor ticks label
    ax.tick_params(axis='both', which='major', labelsize=3)
    ax.tick_params(axis='both', which='minor', labelsize=1)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.savefig(s_path, dpi=1024)
    return ax

def add_cm_to_tb(name):
    png_file = name
    image = Image.open(png_file)
    image_tensor = F.to_tensor(image)
    return image_tensor

def plot_confusion_matrix_diagonal(conf_matrix,classes,name):
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    diagonal = np.diag(conf_matrix)
    y_pos = range(len(diagonal))
    plt.tick_params(axis='x', which='major', labelsize=3)
    plt.tick_params(axis='x', which='minor', labelsize=1)
    plt.bar(y_pos, diagonal)
    plt.xticks(y_pos, classes, color='orange', rotation=90, horizontalalignment='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Confusion Matrix - Diagonal Only')
    plt.tight_layout()
    plt.savefig(name, dpi=1024)

 
