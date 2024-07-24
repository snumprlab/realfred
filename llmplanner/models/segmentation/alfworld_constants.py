from collections import OrderedDict

########################################################################################################################
# General Settings

DEBUG = True
EVAL = False
LOG_FILE = 'logs_gen'

RECORD_VIDEO_IMAGES = True
RECORD_SMOOTHING_FACTOR = 1
DATA_SAVE_PATH = "dataset/new_trajectories"

OPEN_LOOP = True
FULL_OBSERVABLE_STATE = True

########################################################################################################################
# Generation Ablations

MAX_NUM_OF_OBJ_INSTANCES = 3     # when randomly initializing the scene, create duplicate instance up to this number
PICKUP_REPEAT_MAX = 4            # how many of the target pickup object to generate in [1, MAX] (randomly chosen)
RECEPTACLE_SPARSE_POINTS = 50    # increment for how many points to leave free for sparsely populated receptacles
RECEPTACLE_EMPTY_POINTS = 200    # increment for how many points to leave free for empty receptacles

MIN_VISIBLE_RATIO = 0.0011       # minimum area ratio (with respect to image size) of visible object
PLANNER_MAX_STEPS = 100          # if the generated plan is more than these steps, discard the traj
MAX_EPISODE_LENGTH = 1000        # maximum number of API steps allowed per trajectory

FORCED_SAMPLING = False          # set True for debugging instead of proper sampling
PRUNE_UNREACHABLE_POINTS = False  # prune navigation points that were deemed unreachable by the proprocessing script

########################################################################################################################
# Goals

GOALS = ["pick_and_place_simple",
         "pick_two_obj_and_place",
         "look_at_obj_in_light",
         "pick_clean_then_place_in_recep",
         "pick_heat_then_place_in_recep",
         "pick_cool_then_place_in_recep",
         "pick_and_place_with_movable_recep"]

GOALS_VALID = {"pick_and_place_simple": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
               "pick_two_obj_and_place": {"Kitchen", "LivingRoom", "Bathroom", "Bedroom"},
               "look_at_obj_in_light": {"LivingRoom", "Bedroom"},
               "pick_clean_then_place_in_recep": {"Kitchen", "Bathroom"},
               "pick_heat_then_place_in_recep": {"Kitchen"},
               "pick_cool_then_place_in_recep": {"Kitchen"},
               "pick_and_place_with_movable_recep": {"Kitchen", "LivingRoom", "Bedroom"}}

pddl_goal_type = "pick_and_place_simple"  # default goal type

########################################################################################################################
# Video Settings

# filler frame IDs
BEFORE = 0
MIDDLE = 1
AFTER = 2

# number of image frames to save before and after executing the specified action
SAVE_FRAME_BEFORE_AND_AFTER_COUNTS = {
    'OpenObject': [2, 0, 2],
    'CloseObject': [2, 0, 2],
    'PickupObject': [5, 0, 10],
    'PutObject': [5, 0, 10],
    'CleanObject': [3, 0, 5],
    'HeatObject': [3, 0, 5],
    'CoolObject': [3, 30, 5],
    'ToggleObjectOn': [3, 0, 15],
    'ToggleObjectOff': [1, 0, 5],
    'SliceObject': [3, 0, 7]
}

# FPS
VIDEO_FRAME_RATE = 5

########################################################################################################################
# Data & Storage

save_path = DATA_SAVE_PATH
data_dict = OrderedDict()  # dictionary for storing trajectory data to be dumped

########################################################################################################################
# Unity Hyperparameters

BUILD_PATH = None
X_DISPLAY = '0'

AGENT_STEP_SIZE = 0.25
AGENT_HORIZON_ADJ = 15
AGENT_ROTATE_ADJ = 90
CAMERA_HEIGHT_OFFSET = 0.75
VISIBILITY_DISTANCE = 1.5
HORIZON_GRANULARITY = 15

RENDER_IMAGE = True
RENDER_DEPTH_IMAGE = True
RENDER_CLASS_IMAGE = True
RENDER_OBJECT_IMAGE = True

MAX_DEPTH = 5000
STEPS_AHEAD = 5
SCENE_PADDING = STEPS_AHEAD * 3
SCREEN_WIDTH = DETECTION_SCREEN_WIDTH = 300
SCREEN_HEIGHT = DETECTION_SCREEN_HEIGHT = 300
MIN_VISIBLE_PIXELS = 10


########################################################################################################################
# Scenes and Objects

TRAIN_SCENE_NUMBERS = list(range(7, 31))           # Train Kitchens (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(207, 231)))  # Train Living Rooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(307, 331)))  # Train Bedrooms (24/30)
TRAIN_SCENE_NUMBERS.extend(list(range(407, 431)))  # Train Bathrooms (24/30)

TEST_SCENE_NUMBERS = list(range(1, 7))             # Test Kitchens (6/30)
TEST_SCENE_NUMBERS.extend(list(range(201, 207)))   # Test Living Rooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(301, 307)))   # Test Bedrooms (6/30)
TEST_SCENE_NUMBERS.extend(list(range(401, 407)))   # Test Bathrooms (6/30)


SCENE_NUMBERS = TRAIN_SCENE_NUMBERS + TEST_SCENE_NUMBERS

# Scene types.
# REALFRED
SCENE_TYPE = {"Entire": range(1, 151),
              "Splitted": range(201, 301)}

# REALFRED
OBJECTS = [
    "AirConditioner",
    "AirPurifier",
    "AlarmClock",
    "Apple",
    "ArmChair",
    "Banana",
    "BaseballBat",
    "Basket",
    "BasketBall",
    "Bathtub",
    "BathtubBasin",
    "Bed",
    "Book",
    "Bottle",
    "Bowl",
    "Box",
    "Bread",
    "BreadSliced",
    "ButterKnife",
    "CD",
    "Cabinet",
    "Caculator",
    "Candle",
    "CannedFood",
    "CellPhone",
    "Chair",
    "Cloth",
    "CoffeeMachine",
    "CoffeeTable",
    "Comb",
    "ComputerMouse",
    "Controller",
    "Cookie",
    "CounterTop",
    "CreditCard",
    "Cup",
    "CuttingBoard",
    "Desk",
    "DeskLamp",
    "Desktop",
    "DiningTable",
    "DishSponge",
    "Drawer",
    "Dresser",
    "DressingTable",
    "Dumbbell",
    "Egg",
    "Eggplant",
    "Fan",
    "Faucet",
    "FloorLamp",
    "Flute",
    "Footstool",
    "Fork",
    "Fridge",
    "GarbageCan",
    "Grape",
    "Hammer",
    "HandTowel",
    "HandTowelHolder",
    "Headband",
    "Kettle",
    "KeyChain",
    "Keyboard",
    "Knife",
    "Ladle",
    "Laptop",
    "Lemon",
    "Lettuce",
    "Mango",
    "Microwave",
    "Mirror",
    "Monitor",
    "Mug",
    "Newspaper",
    "Onion",
    "Ottoman",
    "Pan",
    "Peach",
    "Pen",
    "Pencil",
    "PencilCase",
    "PepperShaker",
    "Pillow",
    "Plate",
    "Plum",
    "Plunger",
    "Pot",
    "Potato",
    "Pumpkin",
    "RemoteControl",
    "Safe",
    "SaltShaker",
    "Scissors",
    "Shelf",
    "ShelvingUnit",
    "ShowerCurtain",
    "ShowerHead",
    "SideTable",
    "Sink",
    "SinkBasin",
    "SoapBar",
    "SoapBottle",
    "Sofa",
    "Spatula",
    "Spoon",
    "SprayBottle",
    "Statue",
    "Stool",
    "StoveBurner",
    "Strawberry",
    "TVStand",
    "Table",
    "Tambourine",
    "Television",
    "TennisRacket",
    "TissueBox",
    "Toilet",
    "ToiletPaper",
    "Tomato",
    "Towel",
    "TowelHolder",
    "Undefined",
    "Vase",
    "WashingMachine",
    "Watch",
    "WateringCan",
    "Watermelon",
    "WineBottle"
]

# REALFRED
OBJECTS_WSLICED = sorted(OBJECTS +
                         ["AppleSliced", "BreadSliced", "LettuceSliced", "EggCracked", "PotatoSliced", "TomatoSliced", "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"])

OBJECTS_LOWER_TO_UPPER = {obj.lower(): obj for obj in OBJECTS}


MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

MOVABLE_RECEPTACLES_SET = set(MOVABLE_RECEPTACLES)
OBJECTS_SET = set(OBJECTS) | MOVABLE_RECEPTACLES_SET

OBJECT_CLASS_TO_ID = {obj: ii for (ii, obj) in enumerate(OBJECTS)}


# REALFRED
RECEPTACLES = {
        'Bathtub',
        'Bowl',
        'Cup',
        'Drawer',
        'Mug',
        'Plate',
        'Shelf',
        'SinkBasin',
        'Box',
        'Cabinet',
        'CoffeeMachine',
        'CounterTop',
        'Fridge',
        'GarbageCan',
        'HandTowelHolder',
        'Microwave',
        'PaintingHanger',
        'Pan',
        'Pot',
        'StoveBurner',
        'DiningTable',
        'CoffeeTable',
        'SideTable',
        'ToiletPaperHanger',
        'TowelHolder',
        'Safe',
        'BathtubBasin',
        'ArmChair',
        'Toilet',
        'Sofa',
        'Ottoman',
        'Dresser',
        'LaundryHamper',
        'Desk',
        'Bed',
        'Cart',
        'TVStand',
        'Toaster',
        'DressingTable',
        'Sink',
        'Table'
    }


RECEPTACLES_SB = set(RECEPTACLES) | {'Sink', 'Bathtub'}
OBJECTS_DETECTOR = (set(OBJECTS_WSLICED) - set(RECEPTACLES_SB)) | set(MOVABLE_RECEPTACLES)
OBJECTS_DETECTOR -= {'Blinds', 'Boots', 'Cart', 'Chair', 'Curtains', 'Footstool', 'Mirror', 'LightSwtich', 'Painting', 'Poster', 'ShowerGlass', 'Window'}
STATIC_RECEPTACLES = set(RECEPTACLES_SB) - set(MOVABLE_RECEPTACLES)

OBJECTS_DETECTOR = sorted(list(OBJECTS_DETECTOR))
STATIC_RECEPTACLES = sorted(list(STATIC_RECEPTACLES))
ALL_DETECTOR = sorted(list(set(OBJECTS_DETECTOR) | set(STATIC_RECEPTACLES)))

# object parents
# OBJ_PARENTS = {obj: obj for obj in OBJECTS}
# OBJ_PARENTS['AppleSliced'] = 'Apple'
# OBJ_PARENTS['BreadSliced'] = 'Bread'
# OBJ_PARENTS['LettuceSliced'] = 'Lettuce'
# OBJ_PARENTS['PotatoSliced'] = 'Potato'
# OBJ_PARENTS['TomatoSliced'] = 'Tomato'

# force a different horizon view for objects of (type, location). If the location is None, force this horizon for all
# objects of that type.
FORCED_HORIZON_OBJS = {
    ('FloorLamp', None): 0,
    ('Fridge', 18): 30,
    ('Toilet', None): 15,
}

# openable objects with fixed states for transport.
FORCED_OPEN_STATE_ON_PICKUP = {
    'Laptop': False,
}

# list of openable classes.
OPENABLE_CLASS_LIST = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe', 'Box']
OPENABLE_CLASS_SET = set(OPENABLE_CLASS_LIST)

########################################################################################################################