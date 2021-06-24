
import torch
import torchvision
from .networks import VisualNet, AudioNet, weights_init
from .audioVisual_model import AudioVisualModel


class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights=''):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = VisualNet(original_resnet)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    # builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        # AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net

    def get_model(self, opt):
        if opt.model == 'audioVisual':
            net_visual = self.build_visual(weights=opt.weights_visual)
            net_audio = self.build_audio(
                ngf=opt.unet_ngf,
                input_nc=opt.unet_input_nc,
                output_nc=opt.unet_output_nc,
                weights=opt.weights_audio)
            nets = (net_visual, net_audio)
            model = AudioVisualModel(nets, opt)
            print('Loaded model: audioVisual')
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)

        return model , nets
