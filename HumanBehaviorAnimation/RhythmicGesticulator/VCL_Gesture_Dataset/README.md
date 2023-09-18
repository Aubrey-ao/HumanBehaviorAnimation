# VCL Gesture Dataset

## Dataset Information

The VCL Gesture Dataset contains 30 sequences, totaling 269.93 minutes of monologues performed by 5 male speakers in Chinese. 80% of the data is allocated for training purposes, while the remaining 20% is reserved for testing. The dataset comprises three types of data, including 3D full-body motion in the .bvh format, speech audio in the .wav format, and aligned transcripts in the .srt format, which are recognized by [Whisper](https://openai.com/research/whisper). 

You can download the dataset in the .zip format from here (comming soon).

Directly detecting verbal stresses, such as audio onsets, as gesture beats is a widely used approach in gesture-related research. However, this approach is not always perfect as gesture beats may not always correspond to stressed syllables [[McClave 1994]](https://link.springer.com/article/10.1007/BF02143175). And accurately modeling complex gestural rhythm remains an open question. To facilitate relevant studies, we manually labeled timestamps of emphasis, namely gesture beats, for speaker01's data, which amounts to 29.27 minutes of speech-gesture data. You can download the speech-gesture data with beat labels in the .zip format from here (comming soon).

## Citation

If you use this dataset, please cite our paper:

```
@article{ao2022rhythmic_gesticulator,
  title={Rhythmic gesticulator: Rhythm-aware co-speech gesture synthesis with hierarchical neural embeddings},
  author={Ao, Tenglong and Gao, Qingzhe and Lou, Yuke and Chen, Baoquan and Liu, Libin},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  pages={1--19},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

## Limitations

* The timestamps of transcripts produced by Whisper are at the utterance-level and can be inaccurate by several seconds. [WhisperX](https://github.com/m-bain/whisperX) can be used to improve the granularity (from utterance-level to word-level) and accuracy of the timestamps.
