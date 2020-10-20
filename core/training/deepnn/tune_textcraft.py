
import sys
import torch
import deprecated
import torch.utils.data as utils
import copy
import torch.nn as nn
from craft_text_detector import Craft

sys.path.append("../")
from core.training.deepnn.train_wrappers import  TrainModel
from core.training.deepnn.train_wrappers import TrainWithFeatureChannels
from core.datahandling.datahandler_img import DatasetImgCraftDefault


@deprecated(version='1.0.0', reason="this method is currently be transferred from notebooks, but will be removed and modularized in the future")
def main ():
    """
    tuning crafttext for object detection
    Returns:

    """
    DIR ="../data/craft_temp/"

    print(torch.cuda.is_available())
    craft = Craft(output_dir=DIR, crop_type="box", cuda=True, text_threshold=.4 ,link_threshold=.4
                  ,low_text=.4)  # just extract network
    base_model =copy.deepcopy(craft.craft_net)

    test_handler =DatasetImgCraftDefault(base_model ,"../data/craft_temp/temp_train")
    model_train =copy.deepcopy(craft.craft_net)


    loader_gen =torch.utils.data.DataLoader(test_handler, batch_size=1, shuffle=False ,num_workers=0)
    lossType =nn.L1Loss(reduction="sum")
    train_obj =TrainModel(model_train ,lossType)
    optimizer =torch.optim.Adam(model_train.parameters() ,lr=.00001 ,weight_decay=.0005)
    # not again fff
    for i in range(20):
        print(train_obj.train(5 ,loader_gen ,optimizer ,True))


if __name__ == '__main__':
    main()