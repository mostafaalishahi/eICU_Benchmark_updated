class Config():
    def __init__(self, args):
        self.seed = 36

        # data dir
        self.root_dir = 'extracted data dir'
        self.eicu_dir = 'eICU dataset dir'

        # task details
        self.task = args.task # ['phen', 'dec', 'mort', 'rlos']
        self.num = args.num #
        self.cat = args.cat #
        self.n_cat_class = 429
        self.k_fold = 5

        # self.regression = False

        #model params
        self.save_dir = 'results/'
        self.embedding_dim = 5
        self.epochs = 100
        self.batch_size = 512

        self.ann = args.ann #
        self.lir = args.lir #
        self.lor = args.lor #

        self.ohe = args.ohe #
        self.mort_window = args.mort_window #48
        self.lr = 0.0001
        self.dropout = 0.3
        self.rnn_layers = 2
        self.rnn_units = [64, 64] # list of numbers


        # decompensation
        self.dec_cat = ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal']
        self.dec_num = ['admissionheight', 'admissionweight', 'age', 'Heart Rate', 'MAP (mmHg)','Invasive BP Diastolic', 'Invasive BP Systolic',
                        'O2 Saturation', 'Respiratory Rate', 'Temperature (C)', 'glucose', 'FiO2', 'pH']


        #phenotyping
        self.col_phe = ["Respiratory failure", "Fluid disorders",
                    "Septicemia", "Acute and unspecified renal failure", "Pneumonia",
                    "Acute cerebrovascular disease",
                    "Acute myocardial infarction", "Gastrointestinal hem", "Shock", "Pleurisy",
                    "lower respiratory", "Complications of surgical", "upper respiratory",
                    "Hypertension with complications", "Essential hypertension", "CKD", "COPD",
                    "lipid disorder", "Coronary athe", "DM without complication",
                    "Cardiac dysrhythmias",
                    "CHF", "DM with complications", "Other liver diseases", "Conduction disorders"]
