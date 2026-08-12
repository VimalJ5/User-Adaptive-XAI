"""
config.py
=========
Central configuration for the XAI verbalization pipeline.
Edit this file to tune behaviour across all modules.
"""

# ---------------------------------------------------------------------------
# LLM generation parameters
# ---------------------------------------------------------------------------
LLM_LOCAL_PATH = "microsoft/phi-2"    # <-- set to your local Qwen directory
LLM_MAX_NEW_TOKENS = 120
MIN_NEW_TOKENS = 40
NUM_BEAMS = 2                       # keep low (2-4) for 6 GB VRAM

# ---------------------------------------------------------------------------
# Task model (loaded from HuggingFace, change per task)
# ---------------------------------------------------------------------------
TASK_MODEL_NAME = "hamzab/roberta-fake-news-classification"   # was "ProsusAI/finbert"
TASK_LABELS = ["fake", "real"]   # verify this order matches your precomputed LIME JSON's predicted_label strings

# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------
DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_allagree"
# DATASET_CONFIG = None
DATASET_SPLIT = "train"             # phrasebank has no official test split
SAMPLE_SIZE = 50                    # <-- change this (10-50 recommended)
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# User category -> lambda mapping
# Higher lambda = stronger readability constraint = simpler output.
# BEGINNER: most constrained, EXPERT: least constrained (near free generation)
# ---------------------------------------------------------------------------
LAMBDA_MAP = {
    "BEGINNER":     4.0,
    "INTERMEDIATE": 1.0,
    "EXPERT":       -2.0,
}

# ---------------------------------------------------------------------------
# Readability penalty weights
# Each weight controls how much that signal contributes to the penalty.
# They are summed and scaled by lambda at generation time.
# ---------------------------------------------------------------------------
HARDNESS_WEIGHTS = {
    "dale_chall":    0.6,   # unfamiliar word penalty (main driver)
    "syllable":      0.4,   # avg syllables/word
    "polysyllabic":  0.3,   # words >= POLYSYLLABIC_THRESHOLD syllables (SMOG signal)
    "char_per_word": 0.2,   # avg chars/word (ARI signal)
    "clause":        0.2,   # subordinate clause markers (syntactic complexity)
    "length":        0.5,   # total word count pressure (nudges toward EOS)
    "sentence_len":  0.3,   # avg words/sentence pressure
}

# ---------------------------------------------------------------------------
# Normalisation caps (used to scale raw counts to [0, 1])
# ---------------------------------------------------------------------------
MAX_LENGTH_CAP        = 80    # word count at which length penalty is maxed
MAX_SYLLABLE_CAP      = 4.0   # syllables/word at which syllable penalty is maxed
CHAR_PER_WORD_CAP     = 10    # chars/word at which char penalty is maxed
SENTENCE_LEN_CAP      = 25    # words/sentence at which sentence-len penalty maxes
POLYSYLLABIC_THRESHOLD = 3    # words with >= this many syllables are "polysyllabic"

# ---------------------------------------------------------------------------
# LIME settings
# ---------------------------------------------------------------------------
LIME_NUM_FEATURES = 6     # top-K tokens returned by LIME
LIME_NUM_SAMPLES  = 500    # perturbation samples (lower = faster, less accurate)

# ---------------------------------------------------------------------------
# Clause markers (syntactic complexity signal)
# Extend this list as needed.
# ---------------------------------------------------------------------------
CLAUSE_MARKERS = {
    "although", "because", "since", "however", "therefore", "whereas",
    "nevertheless", "furthermore", "consequently", "notwithstanding",
    "despite", "unless", "whereby", "herein", "therein", "albeit",
}

# ---------------------------------------------------------------------------
# Domain-specific whitelist
# Words in this set are EXEMPT from dale-chall / syllable penalties.
#   EXPERT       : fully exempt (no penalty)
#   INTERMEDIATE : 50% penalty reduction
#   BEGINNER     : no exemption (full penalty applies)
# Fill this in per task/domain. Empty = no exemptions.
# ---------------------------------------------------------------------------
DOMAIN_WHITELIST: set = {
    # --- finance task: add domain-critical terms here ---
    # e.g. "revenue", "equity", "dividend", "ebitda"
}

# ---------------------------------------------------------------------------
# Dale-Chall familiar word list (~3000 words known to 4th graders)
# PLACEHOLDER: paste the full list from:
# https://www.readabilityformulas.com/articles/dale-chall-readability-word-list.php
# A small working subset is included below so the code runs immediately.
# ---------------------------------------------------------------------------
DALE_CHALL_FAMILIAR: set = {
    "all","another","any","both","each","either","every","few","many","much","neither","none",
    "other","own","same","several","some","us","what","which","who","whom","whose",
    "zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven",
    "twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
    "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety","hundred","thousand",
    "million","billion","first","second","third","fourth","fifth","sixth","seventh","eighth",
    "ninth","tenth","once","twice",
    "black","blue","brown","gold","gray","green","orange","pink","purple","red","silver","tan",
    "white","yellow",
    "baby","boy","brother","child","children","dad","daughter","family","father","friend","girl",
    "grandma","grandpa","grandfather","grandmother","husband","kid","man","men","mom","mother",
    "neighbor","parent","people","person","sister","son","teacher","wife","woman","women",
    "act","add","allow","ask","back","become","bring","build","buy","call","came","carry",
    "cause","change","check","choose","clean","close","come","cook","copy","count","cover",
    "create","cut","decide","develop","draw","drink","drive","drop","eat","end","enjoy","enter",
    "explain","fall","feel","fight","fill","find","follow","forget","get","give","go","grow",
    "happen","hear","help","hit","hold","hope","include","keep","kill","know","lead","learn",
    "leave","let","like","listen","live","look","lose","love","make","mean","meet","miss","move",
    "need","open","play","point","put","read","remain","remove","run","say","see","seem","send",
    "set","show","sit","sleep","speak","stand","start","stay","stop","study","take","talk",
    "teach","tell","think","try","turn","use","wait","walk","want","watch","work","write",
    "able","afraid","ago","alive","alone","already","always","bad","beautiful","better","big",
    "bright","busy","careful","certain","clear","common","dark","dead","dear","deep","different",
    "difficult","early","easy","enough","even","ever","fair","far","fast","fine","free","full",
    "good","great","happy","hard","heavy","high","hot","huge","important","large","last","late",
    "left","light","little","long","low","main","new","nice","normal","now","often","only",
    "open","poor","possible","pretty","quick","quiet","ready","real","right","round","safe",
    "short","simple","slow","small","smart","soft","soon","sorry","special","still","strange",
    "strong","sure","sweet","tall","true","warm","wide","wrong","young",
    "age","air","animal","answer","area","arm","ball","bed","bottom","box","break","bus","car",
    "care","city","class","color","corner","country","course","cup","day","door","eye","eyes",
    "face","fact","field","fire","floor","food","foot","game","ground","group","hand","head",
    "home","hour","house","idea","job","land","letter","life","line","list","matter","minute",
    "money","month","morning","name","nature","night","number","order","page","part","party",
    "place","plan","plant","power","problem","question","reason","road","room","school","sea",
    "side","size","sky","sound","state","story","street","sun","table","thing","thought","time",
    "top","town","tree","type","view","voice","water","way","week","window","word","world","year",
    "above","across","again","almost","along","around","away","below","down","else","here",
    "instead","just","later","maybe","near","never","next","off","out","outside","perhaps",
    "please","quite","rather","since","so","somehow","sometimes","then","through","together",
    "too","toward","until","up","upon","usually","very","well","when","where","yet",
    "arms","bone","bones","chest","ear","ears","fat","feet","hair","hands","hip","jaw","joint",
    "joints","knee","knees","leg","legs","lip","lips","mouth","neck","nose","organ","organs",
    "rib","ribs","shoulder","skin","spine","stomach","throat","toe","toes","tooth","teeth",
    "vein","veins","wrist","healthy","sick","ill","diet","drug","drugs","dose","treat",
    "treatment","doctor","nurse","patient","surgery","hospital","studies","result","results",
    "causes","effect","effects","level","levels","rate","rates","types","sign","signs","stage",
    "stages",
    "action","amount","base","based","basic","best","bit","cases","control","current","data",
    "date","days","decision","direct","example","exist","factor","focus","form","found","given",
    "goal","health","human","increase","individual","initial","input","issue","key","less",
    "likely","link","local","lower","major","manage","means","medical","method","model","output",
    "overall","past","patients","period","primary","process","provide","range","reach","recent",
    "reduce","related","report","research","role","sample","serve","share","similar","single",
    "specific","step","support","system","term","test","total","understand","unit","value",
    "various","within","three"
    # plain financial terms safe for beginners
    "stock", "price", "market", "company", "rate", "loss", "gain", "profit",
    "share", "fund", "bank", "cost", "sale", "buy", "sell", "trade", "money",
    "value", "growth", "fall", "rise", "report", "result", "plan", "deal",
}

# ---------------------------------------------------------------------------
# Explanation method switch
# ---------------------------------------------------------------------------
EXPLANATION_METHOD = "sv"   # "cd" (constrained decoding) | "sv" (steering vectors) | "sv_cd"

# ---------------------------------------------------------------------------
# Steering vector (SV) settings — only used when EXPLANATION_METHOD == "sv"
# ---------------------------------------------------------------------------
STEERING_VECTOR_PATH = "v_steering.npy"
STEERING_LAYER = 16
STEERING_SCALE_FRACTION = 0.05   # matches your notebook's final chosen value

STEERING_ALPHA_MAP = {
    "beginner":     -0.8,
    "intermediate":  0.0,
    "expert":        2.0,
}

SV_MAX_NEW_TOKENS = 120
SV_TEMPERATURE = 0.7
SV_TOP_P = 0.9
SV_REPETITION_PENALTY = 1.1

# Precomputed classification + LIME attributions (produced by a prior CD run)
LIME_JSON_PATH = "financial_phrasebank_lime.json"
LOG_FILE_PATH = f"run_log_{EXPLANATION_METHOD}.txt"