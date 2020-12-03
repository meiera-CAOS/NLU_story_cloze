import matplotlib.pyplot as plt
import extract_data

# 3HL, batch_size=32, 10splits cv

epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

cv_NC_normal, cv_LS_normal, cv_FC_normal = extract_data.get_10split_normal_data(epochs=epochs)
cv_NC_dropout, cv_LS_dropout, cv_FC_dropout = extract_data.get_10split_dropout_data(epochs=epochs)

nocv_NC_normal, nocv_LS_normal, nocv_FC_normal = extract_data.get_no_crossvalidate_normal_data(epochs=epochs)
nocv_NC_dropout, nocv_LS_dropout, nocv_FC_dropout = extract_data.get_no_crossvalidate_dropout_data(epochs=epochs)

# ################################# CV

# comparison cv NC drop vs. normal
def plot_cv_NC(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, cv_NC_normal)
    plt.plot(epochs, cv_NC_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('10fold Crossvalidation:')
    plt.legend(['NC_nothing', 'NC_dropscale'])
    if save:
        plt.savefig('plots/cv_NC.png')
    else:
        plt.show()


# comparison cv LS drop vs. normal
def plot_cv_LS(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, cv_LS_normal)
    plt.plot(epochs, cv_LS_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('10fold Crossvalidation:')
    plt.legend(['LS_nothing', 'LS_dropscale'])
    if save:
        plt.savefig('plots/cv_LS.png')
    else:
        plt.show()


# comparison cv FC drop vs. normal
def plot_cv_FC(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, cv_FC_normal)
    plt.plot(epochs, cv_FC_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('10fold Crossvalidation:')
    plt.legend(['FC_nothing', 'FC_dropscale'])
    if save:
        plt.savefig('plots/cv_FC.png')
    else:
        plt.show()


# cv dropout comparison NC vs. LS vs. FC
def plot_cv_dropout(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, cv_NC_dropout)
    plt.plot(epochs, cv_LS_dropout)
    plt.plot(epochs, cv_FC_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('10fold Crossvalidation:')
    plt.legend(['NC_dropscale', 'LS_dropscale', 'FC_dropscale'])
    if save:
        plt.savefig('plots/cv_dropout.png')
    else:
        plt.show()


# cv normal comparison NC vs. LS vs. FC
def plot_cv_normal(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, cv_NC_normal)
    plt.plot(epochs, cv_LS_normal)
    plt.plot(epochs, cv_FC_normal)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('10fold Crossvalidation:')
    plt.legend(['NC_normal', 'LS_normal', 'FC_normal'])
    if save:
        plt.savefig('plots/cv_normal.png')
    else:
        plt.show()


# create and save all plots of plot_cv_* functions
def cv_saveall():
    plot_cv_NC(True)
    plt.figure()
    plot_cv_LS(True)
    plt.figure()
    plot_cv_FC(True)
    plt.figure()
    plot_cv_dropout(True)
    plt.figure()
    plot_cv_normal(True)

#cv_saveall()

# ################################# NO_CV

# comparison cv NC drop vs. normal
def plot_nocv_NC(save=False):
    plt.plot(epochs, nocv_NC_normal)
    plt.plot(epochs, nocv_NC_dropout)
    plt.ylim([0.5,0.8])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('No_Crossvalidate')
    plt.legend(['NC_nothing', 'NC_dropscale'])
    if save:
        plt.savefig('plots/nocv_NC.png')
    else:
        plt.show()


# comparison cv LS drop vs. normal
def plot_nocv_LS(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, nocv_LS_normal)
    plt.plot(epochs, nocv_LS_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('No_Crossvalidate')
    plt.legend(['LS_nothing', 'LS_dropscale'])
    if save:
        plt.savefig('plots/nocv_LS.png')
    else:
        plt.show()


# comparison cv FC drop vs. normal
def plot_nocv_FC(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, nocv_FC_normal)
    plt.plot(epochs, nocv_FC_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('No_Crossvalidate')
    plt.legend(['FC_nothing', 'FC_dropscale'])
    if save:
        plt.savefig('plots/nocv_FC.png')
    else:
        plt.show()


# cv dropout comparison NC vs. LS vs. FC
def plot_nocv_dropout(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, nocv_NC_dropout)
    plt.plot(epochs, nocv_LS_dropout)
    plt.plot(epochs, nocv_FC_dropout)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('No_Crossvalidate')
    plt.legend(['NC_dropscale', 'LS_dropscale', 'FC_dropscale'])
    if save:
        plt.savefig('plots/nocv_dropout.png')
    else:
        plt.show()


# cv normal comparison NC vs. LS vs. FC
def plot_nocv_normal(save=False):
    plt.ylim([0.5,0.8])
    plt.plot(epochs, nocv_NC_normal)
    plt.plot(epochs, nocv_LS_normal)
    plt.plot(epochs, nocv_FC_normal)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('No_Crossvalidate')
    plt.legend(['NC_normal', 'LS_normal', 'FC_normal'])
    if save:
        plt.savefig('plots/nocv_normal.png')
    else:
        plt.show()



# create and save all plots of plot_cv_* functions
def nocv_saveall():
    plot_nocv_NC(True)
    plt.figure()
    plot_nocv_LS(True)
    plt.figure()
    plot_nocv_FC(True)
    plt.figure()
    plot_nocv_dropout(True)
    plt.figure()
    plot_nocv_normal(True)

cv_saveall()
#plot_nocv_NC()