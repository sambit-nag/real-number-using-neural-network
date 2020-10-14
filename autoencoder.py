import numpy as np
import torch as pt
import os.path
import time

seed = 2020
np.random.seed(seed)
pt.manual_seed(seed)

def random_real_numbers(batch_size=3, place_value=3,low=0,high=99):
    '''
    Real number = Whole number + Decimal number
    
    Inputs
    batch_size: batch_size
    place_value: number of decimals
    
    Outputs:
    whole: numpy array of whole numbers. size(batch_size)
    decimal: numpy array of decimal numbers. size(batch_size)
    '''
    whole = np.random.random_integers(low,high,size=(batch_size))
    decimal = np.random.random(batch_size)
    decimal = np.array([round(i,(np.random.randint(place_value)+1)) for i in decimal.tolist()])
    return whole, decimal


def get_real_num_matrix(whole,decimal,batch_size=10, max_whole_len=2,max_decimal_len=3):
    '''
    Real number = Whole number + Decimal number
    
    Inputs
    whole: whole number value for creating real number
    decimal: decimal number value for creating real number
    batch_size: batch size
    max_whole_len: maximum length of whole number eg 2 for 10.023
    max_decimal_len: maximum length of decimal number eg 3 for 10.023
    
    Outputs
    integer_matrix: matrix of all integers in a real number. size(batch_size,max_whole_len+max_decimal_len,10)
    decimal_matrix: location of decimal in a real number. size(batch_size,max_decimal_len+1)
    '''
    number_of_decimals = [(len(str(i))-2) for i in decimal.tolist()]
    real = whole + decimal
    full_int = np.array([int(i*(10**j)) for i,j in zip(real,number_of_decimals)])
    
    integer_matrix = np.zeros((batch_size,max_whole_len+max_decimal_len,10))
    for i,j in zip(integer_matrix,full_int):
        int_list = [int(k) for k in str(j).zfill(max_whole_len+max_decimal_len)]
        i[np.arange(len(i)), int_list] = 1
    
    decimal_matrix = np.zeros((batch_size,place_value+1))
    decimal_matrix[np.arange(len(decimal_matrix)), number_of_decimals] = 1
    
    return integer_matrix, decimal_matrix

def prediction(sign, integer,decimal):
    arr = []
    for i,j,k in zip(sign,integer,decimal):
        sign_val = np.argmax(i)
        int_val = np.argmax(j,1)
        int_val  =int("".join(map(str, int_val)))
        dec_val = int(np.argmax(k))
        
        real = ((-1)**sign_val) * int_val * (10**-dec_val)
        arr.append(real)
    return arr

def difference_loss(output,target):
    target = pt.argmax(target,dim=2)
    output = pt.argmax(output,dim=2)
    place_holder = pt.tensor([10**4,10**3,10**2,10*1,1])
    
    loss = (pt.abs((output @ place_holder - target @ place_holder).float().mean()))/10**4
    return loss


class Model(pt.nn.Module):
    def __init__(self):
        super().__init__()
        # Input Heads 
        #self.inp_int  = pt.nn.Conv1d(in_channels=5, out_channels=2, kernel_size=2) #(bs,5,10)
        self.inp_int  = pt.nn.Linear(50,18)
        self.inp_decimal = pt.nn.Linear(4,18) #(bs,4)
        self.inp_sign = pt.nn.Linear(2,18) #(bs,2)
        
        # Body 
        self.l1 = pt.nn.Linear(54,22) 
        self.l2 = pt.nn.Linear(22,11)
        self.l3 = pt.nn.Linear(11,50)
    
        
        # Output Heads
        self.out_int = pt.nn.Linear(5*10,50)
        #self.out_int = pt.nn.Conv1d(in_channels=5, out_channels=5, kernel_size=1) #(bs,5,10)
        self.out_decimal = pt.nn.Linear(5*10,4) #(bs,4)
        self.out_sign = pt.nn.Linear(5*10,2) #(bs,2)
        
        
        # Activtions
        self.relu = pt.nn.ReLU()
        self.soft = pt.nn.Softmax(dim=2)
        self.dropout = pt.nn.Dropout(0.05)
    def input_heads(self,x,y,z):
    
        o1 = self.inp_sign(x)
        o1 = self.relu(o1)
        
        
        #o2 = self.inp_int(y)
        o2 = self.inp_int(y.view(self.bs,5*10))
        o2 = self.relu(o2)
        #o2 = o2.view(self.bs,18)
        
        o3 = self.inp_decimal(z)
        o3 = self.relu(o3)
        
        return pt.cat((o1, o2, o3), 1)
    def body(self,x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.l3(x)
        return self.relu(x)
    def forward(self,x,y,z):
        '''
        x,o1: sign matrix
        y,o2: integer matrix
        z,o3: decimal matrix
        '''
        self.bs = x.shape[0] # batch_size
        o = self.input_heads(x,y,z)
        o = self.body(o)
        
        
        o1 = self.out_sign(o)
        o1 = pt.sigmoid(o1)
        
        o2 = self.out_int(o)
        #o2 = self.soft(o2)
        o2 = pt.sigmoid(o2.view(self.bs,5,10))
        
        o3 = self.out_decimal(o)
        o3 = pt.sigmoid(o3)
    
        return o1,o2,o3




batch_size =10
place_value = 3
whole_len= 2
decimal_len = 3

PATH = "/user/dir/path/model.pt" # please provide your path here

model = Model().double()

try:
    if os.path.isfile(PATH):
        model.load_state_dict(pt.load(PATH))
except:
    os.system("rm "+PATH)

criterion = pt.nn.BCELoss() 
opt = pt.optim.Adam(model.parameters(), lr=0.001)

for i in range(100000):
    
    opt.zero_grad()
    whole, decimal = random_real_numbers(batch_size,place_value)
    int_matrix, dec_matrix = get_real_num_matrix(whole,decimal)

    sign = np.zeros((batch_size,2))
    if i%2==0:
        sign[:,0] = 1
    else:
        sign[:,1] = 1

    sign = pt.from_numpy(sign)
    int_matrix = pt.from_numpy(int_matrix)
    dec_matrix = pt.from_numpy(dec_matrix)
    
    o1,o2,o3 = model(sign,int_matrix,dec_matrix)
    

    loss1 = criterion(o1, sign)
    loss2 = criterion(o2, int_matrix)
    loss3 = criterion(o3, dec_matrix)
    loss4 = difference_loss(o2, int_matrix)
    
    loss = loss1 + loss2 + loss3 + loss4
    loss.backward()
    pt.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
    opt.step()
    if i%1000==0:
        print("\n Epoch: ",i)
        print("Real: ",whole+decimal)
        print("Pred: ",prediction(o1.detach().numpy(),o2.view(batch_size,5,10).detach().numpy(),o3.detach().numpy()))
        #print("Pred: ",prediction(o1.detach().numpy(),int_matrix.view(batch_size,5,10).detach().numpy(),o3.detach().numpy()))
        print("Loss: ",loss.item())
        pt.save(model.state_dict(), PATH)

    if i%5000==0 and i!=0:
        #time.sleep(20)
        pass



class CNNModel(pt.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input Heads 
        self.inp_int  = pt.nn.Conv1d(in_channels=5, out_channels=2, kernel_size=2) #(bs,5,10)
        self.inp_bn = pt.nn.BatchNorm1d(2)
        self.inp_decimal = pt.nn.Linear(4,18) #(bs,4)
        self.inp_sign = pt.nn.Linear(2,18) #(bs,2)
        
        # Body 
        self.conv1 = pt.nn.Conv1d(in_channels=6, out_channels=3, kernel_size=2)
        self.bn1 = pt.nn.BatchNorm1d(3)
        self.conv2 = pt.nn.Conv1d(in_channels=3, out_channels=1, kernel_size=2)
        self.bn2 = pt.nn.BatchNorm1d(1)
        self.conv3 = pt.nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2,padding=1)
        self.bn3 = pt.nn.BatchNorm1d(3)
        self.conv4 = pt.nn.Conv1d(in_channels=3, out_channels=6, kernel_size=2,padding=1)
        self.bn4 = pt.nn.BatchNorm1d(6)
        
        # Output Heads
        self.out_int = pt.nn.Conv1d(in_channels=6, out_channels=5, kernel_size=2,padding=1) #(bs,5,10)
        self.out_decimal = pt.nn.Linear(6*9,4) #(bs,4)
        self.out_sign = pt.nn.Linear(6*9,2) #(bs,2)
        
        # Dropout
        self.drop = pt.nn.Dropout(0.5)
        
        # Activtions
        self.relu = pt.nn.ReLU()
        self.soft1 = pt.nn.Softmax(dim=1)
        self.soft2 = pt.nn.Softmax(dim=2)
        
    def input_heads(self,x,y,z):
        o1 = self.inp_sign(x)
        o1 = o1.view(self.bs,2,9)
        o1 = self.relu(o1)
        
        o2 = self.inp_int(y)
        o2 = self.relu(o2)
        o2 = self.inp_bn(o2)
        
        o3 = self.inp_decimal(z)
        o3 = o3.view(self.bs,2,9)
        o3 = self.relu(o3)
        
        return pt.cat((o1, o2, o3), 1)
        
    def body(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return self.relu(x)
        
    def forward(self,x,y,z):
        '''

        # x,o1: sign matrix
        # y,o2: integer matrix
        # z,o3: decimal matrix

        '''
        self.bs = x.shape[0] # batch_size
        o = self.input_heads(x,y,z)
        o = self.body(o)
        
        o1 = self.out_sign(o.reshape(self.bs,6*9))
        o1 = self.soft1(o1)
        
        o2 = self.out_int(o)
        o2 = self.soft2(o2)
        
        o3 = self.out_decimal(o.reshape(self.bs,6*9))
        o3 = self.soft1(o3)
        return o1,o2,o3

