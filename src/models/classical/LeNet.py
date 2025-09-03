# Some of the code taken from
# https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from models.BaseModel import BaseModel
from models.ModelFactory import ModelFactory

@ModelFactory.model("lenet")
class LeNet(BaseModel):
	def __init__(self, **kwargs):
		# Call the parent constructor
		super().__init__(**kwargs)

		# Initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=self.input_size[0], out_channels=16, kernel_size=(5, 5), padding=2)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

		# Initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2)
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		# Initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=1568, out_features=500)
		self.relu3 = ReLU()

		# Initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=self.output_size)
		self.logSoftmax = LogSoftmax(dim=1)

	def forward(self, inputs):
		# ==== Convolutional layer
		logits = self.conv1(inputs)
		logits = self.relu1(logits)
		logits = self.maxpool1(logits)
		logits = self.conv2(logits)
		logits = self.relu2(logits)
		logits = self.maxpool2(logits)

		logits = flatten(logits, 1) # flatten the output from the previous layer and pass it

		# ==== FC layer
		logits = self.fc1(logits)
		logits = self.relu3(logits)

		# ==== Softmax classifier
		logits = self.fc2(logits)
		output = self.logSoftmax(logits)

		return output # return the output predictions