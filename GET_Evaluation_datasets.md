imagenet-1k -> https://www.kaggle.com/datasets/sautkin/imagenet1kvalid/code

objectnet -> https://huggingface.co/datasets/timm/objectnet?viewer_api=true
``` bash
curl -X GET \
     -H "Authorization: Bearer $HF_TOKEN" \
     "https://datasets-server.huggingface.co/rows?dataset=timm%2Fobjectnet&config=default&split=test&offset=0&length=100"
```

imagenet-v2 -> https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main