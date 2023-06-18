# tweet-robert

![Roberta (1)](https://github.com/ugiugi0823/tweet-sa-robert/assets/106899647/64cc0917-2ae9-4a90-9db7-52b6c8724320)



## 코랩으로 쉽게 돌려보기!!


구글 드라이브 연결하기

```
from google.colab import drive
drive.mount('/content/drive')
```

깃 허브 레포 가져오기
```
!git clone https://github.com/ugiugi0823/tweet-sa-robert.git
%cd tweet-sa-robert
```

```
🔥🔥 무조건, .sh 파일을 수정해야 해요!, 허깅페이스 토큰, wandb 토큰을 입력해야지만, 정상적으로 돌아갑니다. 🔥🔥
```

감성 분석 모델을 **훈련** 시키고 싶다면!!!
```
!bash train.sh
```


감성 분석 모델을 **테스트** 시키고 싶다면!!!
```
!bash test.sh
```
