# Yolo Implementation
Implementation of Yolov1 using Tensorflow 2.0

## loss.py
* 모델의 loss function을 정의

## model.py
* Keras Subclassing형태의 모델 구조 class 정의

## dataset.py
* 데이터 전처리 및 batch 단위로 묶는 로직

## utils.py
* 딥러닝 메인 로직 외에 유틸리티성 기능을 모아놓은 로직

## train.py
모델 클래스를 인스턴스로 선언하고
For loop을 돌면서 gradient descent를 수행하면서
파라미터를 업데이터하는 로직

## evaluate.py
Training된 파라미터를 불러와서 evaluation이나 test/inference를 진행하는 로직



