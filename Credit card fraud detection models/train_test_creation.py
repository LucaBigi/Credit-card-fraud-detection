from functions import *
seme=20
np.random.seed(seme)
torch.manual_seed(seme)
random.seed(seme)

INPUT_FILE = "creditcard.csv"
OUTPUT_TRAIN = "credit_card_train.csv"
OUTPUT_TEST = "credit_card_test.csv"

df = pd.read_csv(INPUT_FILE)

train, test = stratified_split(INPUT_FILE, seme, test_size=0.2)
train.to_csv(OUTPUT_TRAIN, index=False)
test.to_csv(OUTPUT_TEST, index=False)