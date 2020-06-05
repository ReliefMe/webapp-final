import joblib
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_file(file_path):
    test_features = []
    
    signal, sr = librosa.load(file_path, sr = 22050)

    n_fft = 2048
    n_mfcc = 13
    hop_length = 512
    num_segments = 3
    SAMPLE_RATE = 22050
    DURATION = 10  # measured in seconds.
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    
    num_samples_per_segment =  int(SAMPLES_PER_TRACK / num_segments)

    for s in range(num_segments):
        start_sample = num_samples_per_segment * s  # if s= 0 -> then start_sample = 0 
        finish_sample = start_sample + num_samples_per_segment 
        
        # features
        rolloff = librosa.feature.spectral_rolloff(y=signal[start_sample: finish_sample], sr=sr, roll_percent=0.1)

        pitches, magnitudes = librosa.piptrack(y=signal[start_sample: finish_sample], sr=sr)

        mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                    sr =sr,
                                    n_fft = n_fft,
                                    n_mfcc = n_mfcc,
                                    hop_length = hop_length
                                    )
        chroma_cq = librosa.feature.chroma_cqt(y=signal[start_sample: finish_sample], sr=sr)

        # Combining all the features
        features = np.concatenate((pitches, rolloff, mfcc, chroma_cq), axis = 0)
        test_features.append(features)
        test_feat = np.array(test_features)
        model_features = test_feat.reshape(test_feat.shape[0], (test_feat.shape[1]*test_feat.shape[2]))
    
    return model_features

# def predict(cough_fp, saved_model_fp):
#     loaded_model = joblib.load(saved_model_fp)
#     cough_features = load_file(cough_fp)
    
#     result = loaded_model.predict_proba(cough_features)
#     print("Results are : ", result)

    
#     class_neg = []
#     class_pos = []
#     l = 0
#     for i in result:
#         j = np.argmax(i)
#         k = result[l][j]
#         if j == 0:
#             class_neg.append(k)
#         else:
#             class_pos.append(k)
#         l += 1

#     print("class neg: ", class_neg)
#     print("class pos: ", class_pos)
#     if not class_neg:
#         print("covid positive")
#         prob_pos = np.mean(class_pos)
#         print("prob posit: ", prob_pos)
# #         return prob_neg
#     elif not class_pos:
#         print("covid negative")
#         prob_neg = np.mean(class_neg)
#         print("prob neg: ", prob_neg)
# #         return prob_pos
#     else:
#         prob_neg = np.mean(class_neg)
#     #     print(m)

#         prob_pos = np.mean(class_pos)
#         if prob_neg > prob_pos:
#             print("covid neg")
#             return "Covid Negatve :" + str(prob_neg)
#         else:
#             print("covid pos")
#             return "Covid Positive" + str(prob_pos)


# ignoring negative and returning 0
# def predict(cough_fp, saved_model_fp):
#     loaded_model = joblib.load(saved_model_fp)
#     cough_features = load_file(cough_fp)
    
#     result = loaded_model.predict_proba(cough_features)
#     print("Results are : ", result)

#     class_neg = []
#     class_pos = []
#     l = 0
#     for i in result:
#         j = np.argmax(i)
#         k = result[l][j]
#         if j == 0:
#             class_neg.append(k)
#         else:
#             class_pos.append(k)
#         l += 1

#     print("class neg: ", class_neg)
#     print("class pos: ", class_pos)
#     if not class_neg:
#         print("covid positive")
#         prob_pos = np.mean(class_pos)
#         print("prob posit: ", prob_pos)
# #         return "Covid positive: " + str(prob_pos)
#         return prob_pos * 100
#     elif not class_pos:
# #         print("covid negative")
# #         prob_neg = np.mean(class_neg)
# #         print("prob neg: ", prob_neg)
# #         return "Covid negative: "+ str(prob_neg)
#         return 0
#     else:
#         prob_neg = np.mean(class_neg)
#     #     print(m)

#         prob_pos = np.mean(class_pos)
#         if prob_neg > prob_pos:
# #             print("covid neg")
# #             return "Covid Negatve :" + str(prob_neg)
#             return 0
#         else:
#             print("covid pos")
# #             return "Covid Positive :" + str(prob_pos)
#             return prob_pos * 100



# returning prob of the class having max vote count
def predict(cough_fp, saved_model_fp):
    loaded_model = joblib.load(saved_model_fp)
    cough_features = load_file(cough_fp)
    
    result = loaded_model.predict_proba(cough_features)
    print("Results are : ", result)

    class_neg = []
    class_pos = []
    l = 0
    for i in result:
        j = np.argmax(i)
        k = result[l][j]
        if j == 0:
            class_pos.append(k)
        else:
            class_neg.append(k)
        l += 1

    print("class neg: ", class_neg)
    print("class pos: ", class_pos)
    if not class_neg:
        print("covid positive")
        prob_pos = np.mean(class_pos)
        print("prob posit: ", prob_pos)
#         return "Covid positive: " + str(prob_pos)
        return prob_pos * 100
    elif not class_pos:
#         print("covid negative")
#         prob_neg = np.mean(class_neg)
#         print("prob neg: ", prob_neg)
#         return "Covid negative: "+ str(prob_neg)
        return 0
    else:
        prob_neg = np.mean(class_neg)
    #     print(m)

        prob_pos = np.mean(class_pos)
        if len(class_neg) > len(class_pos):
            return 0
#         if prob_neg > prob_pos:
#             print("covid neg")
#             return "Covid Negatve :" + str(prob_neg)
#             return 0
        else:
            return prob_pos