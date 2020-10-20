

import sys
import torch
import deprecated
import torch.utils.data as utils
import copy
import torch.nn as nn
from craft_text_detector import Craft

sys.path.append("../")
from core.training.deepnn.train_wrappers import  TrainModel
from core.datahandling.datahandler_img import DatasetImgCraftDefault

from core.models.deepnn import craft_with_coord
def main ():

    DIR="../data/craft_temp/"

    print(torch.cuda.is_available())
    craft = Craft(output_dir=DIR, crop_type="box", cuda=True, text_threshold=.4,link_threshold=.4,low_text=.4) #just extract network
    base_model=copy.deepcopy(craft.craft_net)

    test_handler=DatasetImgCraftDefault(base_model,"../data/craft_temp/temp_train")

    model_train=copy.deepcopy(craft.craft_net)
    craft_coord_ob=craft_with_coord.CraftCoordSimple(model_train)

    loader_gen=torch.utils.data.DataLoader(test_handler, batch_size=1, shuffle=False,num_workers=0)
    lossType=nn.L1Loss(reduction="sum")
    train_obj=TrainModel(craft_coord_ob,lossType)
    optimizer=torch.optim.Adam(craft_coord_ob.parameters(),lr=.00001,weight_decay=.0005)

    for i in range(20):
        print(train_obj.train(5,loader_gen,optimizer,True))


if __name__ == '__main__':
    main()