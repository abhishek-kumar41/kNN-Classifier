import numpy
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def classification_accuracy():

    train_first_formant = []
    train_second_formant = []
    train_class_formant = []
    train_13d_mfcc = []
    train_class_13d_mfcc = []
    train_12d_mfcc = []
    train_class_12d_mfcc = []
    for i in range(48):

        y, fs = librosa.load("train" + str(i+1) + ".wav", sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)
        mfcc_12d = mfcc[1:13]
        mfcc = numpy.transpose(mfcc)

        height, width = mfcc.shape

        for k in range(height):
            train_13d_mfcc.append(list(mfcc[k, :]))
            train_class_13d_mfcc.append(int(i/8)+1)

        mfcc_12d = numpy.transpose(mfcc_12d)
        height1, width1 = mfcc_12d.shape

        for k in range(height1):
            train_12d_mfcc.append(list(mfcc_12d[k, :]))
            train_class_12d_mfcc.append(int(i/8) + 1)

        formant = open("train" + str(i+1) + ".txt", "r").read()
        formant = string_to_matrix(formant)
        formant_ = formant[1:len(formant)-4]

        for j in range(len(formant_)):
            x = formant_[j][0].split(" ")
            train_first_formant.append(float(x[1]))
            train_second_formant.append(float(x[2]))
            train_class_formant.append(int(i/8)+1)

    train_formant_2d = np.zeros(shape=(len(train_first_formant), 2), dtype=float)
    train_formant_2d[:, 0] = train_first_formant
    train_formant_2d[:, 1] = train_second_formant
    train_class_formant = np.array(train_class_formant)

    test_first_formant = []
    test_second_formant = []
    test_class_formant = []
    test_13d_mfcc = []
    test_class_13d_mfcc = []
    test_12d_mfcc = []
    test_class_12d_mfcc = []
    griffin_test_13d_mfcc = []
    griffin_test_class_13d_mfcc = []

    for i in range(12):
        y, fs = librosa.load("test" + str(i + 1) + ".wav", sr=None)
        spectrogram = np.abs(librosa.stft(y))
        y_inv = librosa.griffinlim(spectrogram)

        mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13)
        griffin_mfcc = librosa.feature.mfcc(y=y_inv, sr=fs, n_mfcc=13)
        mfcc_12d = mfcc[1:13]
        mfcc = np.transpose(mfcc)
        griffin_mfcc = np.transpose(griffin_mfcc)
        height, width = mfcc.shape

        for k in range(height):
            test_13d_mfcc.append(list(mfcc[k, :]))
            test_class_13d_mfcc.append(int(i/2)+1)
            griffin_test_13d_mfcc.append(list(griffin_mfcc[k, :]))
            griffin_test_class_13d_mfcc.append((int(i/2) + 1))

        mfcc_12d = np.transpose(mfcc_12d)
        height1, width1 = mfcc_12d.shape

        for k in range(height1):
            test_12d_mfcc.append(list(mfcc_12d[k, :]))
            test_class_12d_mfcc.append(int(i/2) + 1)

        formant = open("test" + str(i + 1) + ".txt", "r").read()
        formant = string_to_matrix(formant)
        formant_ = formant[1:len(formant) - 4]

        for j in range(len(formant_)):
            x = formant_[j][0].split(" ")
            test_first_formant.append(float(x[1]))
            test_second_formant.append(float(x[2]))
            test_class_formant.append(int(i/2) + 1)

    test_formant_2d = np.zeros(shape=(len(test_first_formant), 2), dtype=float)
    test_formant_2d[:, 0] = test_first_formant
    test_formant_2d[:, 1] = test_second_formant
    test_class_formant = np.array(test_class_formant)

    K = [1, 5, 10, 50, 100, 500, 1000]

    Accuracy_formant = accuracy_list(train_formant_2d, test_formant_2d, train_class_formant, test_class_formant)
    Accuracy_13d_mfcc = accuracy_list(train_13d_mfcc, test_13d_mfcc, train_class_13d_mfcc, test_class_13d_mfcc)
    Accuracy_12d_mfcc = accuracy_list(train_12d_mfcc, test_12d_mfcc, train_class_12d_mfcc, test_class_12d_mfcc)

    # Accuracy_griffin_13d_mfcc = accuracy_list(train_13d_mfcc, griffin_test_13d_mfcc, train_class_13d_mfcc,
    #                                           griffin_test_class_13d_mfcc)



    print(f"Classification Accuracy using 2-dimensional first and second formants as feature vector is:")

    print(f"For K = 1: {Accuracy_formant[0]*100} %")
    print(f"For K = 5: {Accuracy_formant[1]*100} %")
    print(f"For K = 10: {Accuracy_formant[2]*100} %")
    print(f"For K = 50: {Accuracy_formant[3]*100} %")
    print(f"For K = 100: {Accuracy_formant[4]*100} %")
    print(f"For K = 500: {Accuracy_formant[5]*100} %")
    print(f"For K = 1000: {Accuracy_formant[6]*100} %\n")

    print(f"Classification Accuracy using 13-dimensional MFCC is as feature vector is:")

    print(f"For K = 1: {Accuracy_13d_mfcc[0] * 100} %")
    print(f"For K = 5: {Accuracy_13d_mfcc[1] * 100} %")
    print(f"For K = 10: {Accuracy_13d_mfcc[2] * 100} %")
    print(f"For K = 50: {Accuracy_13d_mfcc[3] * 100} %")
    print(f"For K = 100: {Accuracy_13d_mfcc[4] * 100} %")
    print(f"For K = 500: {Accuracy_13d_mfcc[5] * 100} %")
    print(f"For K = 1000: {Accuracy_13d_mfcc[6] * 100} %\n")

    print(f"Classification Accuracy using 12-dimensional MFCC is as feature vector is:")

    print(f"For K = 1: {Accuracy_12d_mfcc[0] * 100} %")
    print(f"For K = 5: {Accuracy_12d_mfcc[1] * 100} %")
    print(f"For K = 10: {Accuracy_12d_mfcc[2] * 100} %")
    print(f"For K = 50: {Accuracy_12d_mfcc[3] * 100} %")
    print(f"For K = 100: {Accuracy_12d_mfcc[4] * 100} %")
    print(f"For K = 500: {Accuracy_12d_mfcc[5] * 100} %")
    print(f"For K = 1000: {Accuracy_12d_mfcc[6] * 100} %\n")


    # print(Accuracy)
    # plt.plot(K, Accuracy, "bo")
    # plt.show()

    K_10_13d_mfcc = KNeighborsClassifier(n_neighbors=10)
    K_10_13d_mfcc.fit(train_13d_mfcc, train_class_13d_mfcc)
    K_10_class_predict_13d_mfcc = K_10_13d_mfcc.predict(test_13d_mfcc)
    print(f"Confusion Matrix which resulted in best classification accuracy is:")
    print(confusion_matrix(test_class_13d_mfcc, K_10_class_predict_13d_mfcc))
    # print(classification_report(test_class_formant, class_predict_formant))
    print("\n")
    print("The vowels which are misclassified are: ")
    for t in range(len(test_class_13d_mfcc)):
        if test_class_13d_mfcc[t] != K_10_class_predict_13d_mfcc[t]:
            print(f"True Class={test_class_13d_mfcc[t]}, Predicted Class = {K_10_class_predict_13d_mfcc[t]}")

    K_10_griffin_13d_mfcc = KNeighborsClassifier(n_neighbors=10)
    K_10_griffin_13d_mfcc.fit(train_13d_mfcc, train_class_13d_mfcc)
    K_10_class_predict_griffin_13d_mfcc = K_10_griffin_13d_mfcc.predict(griffin_test_13d_mfcc)
    Accuracy_griffin_13d_mfcc = accuracy(K_10_class_predict_griffin_13d_mfcc, griffin_test_class_13d_mfcc)
    print(f"Accuracy after reconstructing signal back from the spectrogram using Griffin_Lim algorithm is: {Accuracy_griffin_13d_mfcc}")
    print(f"Confusion Matrix using reconstructed signal back from the spectrogram using Griffin_Lim algorithm is:")
    print(confusion_matrix(griffin_test_class_13d_mfcc, K_10_class_predict_griffin_13d_mfcc))
    print("\n")
    print("The vowels which are misclassified are: ")
    for t in range(len(griffin_test_class_13d_mfcc)):
        if griffin_test_class_13d_mfcc[t] != K_10_class_predict_griffin_13d_mfcc[t]:
            print(f"True Class={griffin_test_class_13d_mfcc[t]}, Predicted Class = {K_10_class_predict_griffin_13d_mfcc[t]}")


    return


def accuracy(class_predict, test_class):
    accuracy_count = 0
    for k in range(len(class_predict)):
        if class_predict[k] == test_class[k]:
            accuracy_count = accuracy_count + 1

    return accuracy_count/len(class_predict)


def accuracy_list(train_formant_2d, test_formant_2d, train_class_formant, test_class_formant):
    K_1_formant = KNeighborsClassifier(n_neighbors=1)
    K_5_formant = KNeighborsClassifier(n_neighbors=5)
    K_10_formant = KNeighborsClassifier(n_neighbors=10)
    K_50_formant = KNeighborsClassifier(n_neighbors=50)
    K_100_formant = KNeighborsClassifier(n_neighbors=100)
    K_500_formant = KNeighborsClassifier(n_neighbors=500)
    K_1000_formant = KNeighborsClassifier(n_neighbors=1000)
    K_1_formant.fit(train_formant_2d, train_class_formant)
    K_5_formant.fit(train_formant_2d, train_class_formant)
    K_10_formant.fit(train_formant_2d, train_class_formant)
    K_50_formant.fit(train_formant_2d, train_class_formant)
    K_100_formant.fit(train_formant_2d, train_class_formant)
    K_500_formant.fit(train_formant_2d, train_class_formant)
    K_1000_formant.fit(train_formant_2d, train_class_formant)
    K_1_class_predict_formant = K_1_formant.predict(test_formant_2d)
    K_5_class_predict_formant = K_5_formant.predict(test_formant_2d)
    K_10_class_predict_formant = K_10_formant.predict(test_formant_2d)
    K_50_class_predict_formant = K_50_formant.predict(test_formant_2d)
    K_100_class_predict_formant = K_100_formant.predict(test_formant_2d)
    K_500_class_predict_formant = K_500_formant.predict(test_formant_2d)
    K_1000_class_predict_formant = K_1000_formant.predict(test_formant_2d)

    Accuracy_formant = [accuracy(test_class_formant, K_1_class_predict_formant),
                        accuracy(test_class_formant, K_5_class_predict_formant),
                        accuracy(test_class_formant, K_10_class_predict_formant),
                        accuracy(test_class_formant, K_50_class_predict_formant),
                        accuracy(test_class_formant, K_100_class_predict_formant),
                        accuracy(test_class_formant, K_500_class_predict_formant),
                        accuracy(test_class_formant, K_1000_class_predict_formant)]

    return Accuracy_formant


def string_to_matrix(string):
    line_split = list(string.split("\n"))
    matrix = []

    for item in line_split:
        line = []
        for data in item.split("   "):
            line.append(data)
        matrix.append(line)

    return matrix


def main():

    classification_accuracy()

    return


if __name__ == '__main__':
    main()
