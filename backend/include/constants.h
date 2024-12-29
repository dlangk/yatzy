#define CATEGORY_COUNT 15

#define CATEGORY_ONES 0
#define CATEGORY_TWOS 1
#define CATEGORY_THREES 2
#define CATEGORY_FOURS 3
#define CATEGORY_FIVES 4
#define CATEGORY_SIXES 5
#define CATEGORY_ONE_PAIR 6
#define CATEGORY_TWO_PAIRS 7
#define CATEGORY_THREE_OF_A_KIND 8
#define CATEGORY_FOUR_OF_A_KIND 9
#define CATEGORY_SMALL_STRAIGHT 10
#define CATEGORY_LARGE_STRAIGHT 11
#define CATEGORY_FULL_HOUSE 12
#define CATEGORY_CHANCE 13
#define CATEGORY_YATZY 14

#define NUM_STATES (64*(1<<15))

#define IS_CATEGORY_SCORED(scored,cat) (((scored)&(1<<(cat)))!=0)
#define SET_CATEGORY(scored,cat) ((scored)|=(1<<(cat)))
#define CLEAR_CATEGORY(scored,cat) ((scored)&=~(1<<(cat)))
#define STATE_INDEX(upper_score, scored_categories) ((upper_score)*(1<<15)+(scored_categories))