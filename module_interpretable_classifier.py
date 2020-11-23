
#import torch
#simport torch.nn as nn

    
#class InterpretableClassifier(nn.Module):
#    def __init__(self, num_channels, num_classes, classifier='fc'):
#        super(InterpretableClassifier, self).__init__()
#        self.classifier = classifier
#        self.fc1 = nn.Linear(num_channels, 200)
#        self.relu1 = nn.ReLU()
#        self.fc2 = nn.Linear(200, 200)
#        self.relu2 = nn.ReLU()
#        self.fc = nn.Linear(200, num_classes)
        
#    def forward(self, x):
#        if self.classifier == 'gap':
#            out = x.mean(1)
#        else:
#            out = self.fc1(x)
#            out = self.relu1(out)
#            out = self.fc2(out)
#            out = self.relu2(out)
#            out = self.fc(out)
 #       return out



import torch
import torch.nn as nn

    
class InterpretableClassifier(nn.Module):
    def __init__(self, num_channels, num_classes, classifier='fc'):
        super(InterpretableClassifier, self).__init__()
        self.classifier = classifier
        
        
        if self.classifier == 'simplefc':
            self.fc_simple = nn.Linear(num_channels, num_classes)
        else:
            self.fc1 = nn.Linear(num_channels, 200)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(200, 200)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(200, num_classes)
            
        
    def forward(self, x):
        if self.classifier == 'gap':
            out = x.mean(1)
            
        elif self.classifier == 'simplefc':
            out = self.fc_simple(x)
        else:
            out = self.fc1(x)
            out = self.relu1(out)
            out = self.fc2(out)
            out = self.relu2(out)
            out = self.fc(out)
        return out
#             out = x.reshape(x.size(0), -1)
#         out = self.drop(out)