
0.	The has two main codes: one for dataset 1, the other for dataset 2
	HW1_Dataset1.py
	HW1_Dataset2.py

1.	To run the code and load the data, someone needs to change the working director path to the folder where the all the data files are unzipped . 

### dataset 1 example ###  
# Reset path
wd=getcwd()
chdir('C:\\Users\HP\\Desktop\\classes\\GT\\HW1')  # need to change the path to run

# Load dataset
x1 = pd.ExcelFile('LendingClub_Data.xls')
data = x1.parse("Cleansed_Data_20180420")

### dataset 2 example ###  
# Reset path
wd=getcwd()
chdir('C:\\Users\HP\\Desktop\\classes\\GT\\HW1')  # TA needs to change this to run the code

# Load dataset
dataRaw = pd.read_csv('Bankloan_Data.csv').drop('Unnamed: 0', axis = 1)
# Loand cleaned data
infile = open('Bankloan_data.obj','rb')


2.	To save running time, some intermediate results are outputted previously as objects; these data objects are submitted along
        along with other datasets as part of homework soluations and will be loaded in the code. 

### dataset 1 example ###  
infile = open('cvResults0.2.obj','rb')
cvResults02 = pickle.load(infile)
infile.close()

### dataset 2 example ###  
infile = open('cvResults.obj02_dataset2','rb')
cvResults02 = pickle.load(infile)
infile.close()


3.	The codes needs to be run sequentially; may take significant amount of time.   
