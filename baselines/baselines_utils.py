from baselines.InternVideo import internvideo
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM, AutoImageProcessor, VideoMAEModel, AutoImageProcessor, CLIPProcessor, CLIPModel

def load_DINOv2_model(model_path):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval().cuda()
    return model, processor


def load_VideoMAE_model(model_path):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = VideoMAEModel.from_pretrained(model_path)
    model.eval().cuda()
    return model, processor


def load_CLIP_model(model_path):
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    model.eval().cuda()
    return model, processor


def load_InternVideo_model(model_path):
    model = internvideo.load_model('baselines/' + model_path + '.ckpt')
    tokenizer = internvideo.tokenize
    model.eval().cuda()
    return model, tokenizer
   