# select 30 wav files for each label

import numpy as np
import csv
import os
from playsound import playsound
import shutil


# parameters
FILES_PER_LABEL = 30


# os.system('clear')
proj_path = os.path.abspath(os.getcwd())
wav_src_path = proj_path + '/data/wav/'
wav_dst_path = proj_path + '/data/selected_wav/'
with open(proj_path + '/csv/categories.csv', newline='') as csvfile:
    categories = np.array(list(csv.reader(csvfile)))
n_categories = len(categories)

# creating target folders
for p in range(n_categories):
    mk_folder_name = wav_dst_path + categories[p, 1]
    os.mkdir(mk_folder_name)
print('Folders created.')

# copy files
for p in range(n_categories):
    label = categories[p, 1]
    print('Processing label: ' + label)
    label_folder = wav_src_path + label
    wav_file_list = os.listdir(label_folder)
    for q in range(FILES_PER_LABEL):
        wav_src_name = label_folder + '/' +  wav_file_list[q]
        print('Copying file #: ' + str(q) + '  ' +wav_file_list[q])
        playsound(wav_src_name)
        wav_dst_name = wav_dst_path + label + '/' +  wav_file_list[q]
        shutil.copy(wav_src_name, wav_dst_name)
print('Done.')


# Note: about 20% of the orignal files has incorrect labels or high noise
# The following set is based on a reduced list of high label quality
# Processing label: Chink and Clink
# Copying file #: 0  212722.wav
# Copying file #: 1  60468.wav
# Copying file #: 2  119133.wav
# Copying file #: 3  15285.wav
# Copying file #: 4  378847.wav
# Copying file #: 5  67041.wav
# Copying file #: 6  31374.wav
# Copying file #: 7  414224.wav
# Copying file #: 8  35631.wav
# Copying file #: 9  368130.wav
# Copying file #: 10  346696.wav
# Copying file #: 11  416285.wav
# Copying file #: 12  348958.wav
# Copying file #: 13  85549.wav
# Copying file #: 14  321290.wav
# Copying file #: 15  419303.wav
# Copying file #: 16  178178.wav
# Copying file #: 17  406716.wav
# Copying file #: 18  416288.wav
# Copying file #: 19  22770.wav
# Copying file #: 20  176226.wav
# Copying file #: 21  404477.wav
# Copying file #: 22  321283.wav
# Copying file #: 23  346701.wav
# Copying file #: 24  212723.wav
# Copying file #: 25  72203.wav
# Copying file #: 26  406717.wav
# Copying file #: 27  126531.wav
# Copying file #: 28  214186.wav
# Copying file #: 29  429234.wav
# Processing label: Clapping Hands
# Copying file #: 0  180762.wav
# Copying file #: 1  73088.wav
# Copying file #: 2  175050.wav
# Copying file #: 3  201655.wav
# Copying file #: 4  192229.wav
# Copying file #: 5  190575.wav
# Copying file #: 6  180759.wav
# Copying file #: 7  404539.wav
# Copying file #: 8  194267.wav
# Copying file #: 9  2081.wav
# Copying file #: 10  404540.wav
# Copying file #: 11  161586.wav
# Copying file #: 12  180754.wav
# Copying file #: 13  351993.wav
# Copying file #: 14  262660.wav
# Copying file #: 15  175042.wav
# Copying file #: 16  231893.wav
# Copying file #: 17  79260.wav
# Copying file #: 18  21181.wav
# Copying file #: 19  175049.wav
# Copying file #: 20  404555.wav
# Copying file #: 21  330296.wav
# Copying file #: 22  61288.wav
# Copying file #: 23  208875.wav
# Copying file #: 24  28225.wav
# Copying file #: 25  266766.wav
# Copying file #: 26  180777.wav
# Copying file #: 27  180779.wav
# Copying file #: 28  257793.wav
# Copying file #: 29  361706.wav
# Processing label: Dropping Coins
# Copying file #: 0  184736.wav
# Copying file #: 1  276173.wav
# Copying file #: 2  335099.wav
# Copying file #: 3  121609.wav
# Copying file #: 4  264987.wav
# Copying file #: 5  336584.wav
# Copying file #: 6  276182.wav
# Copying file #: 7  181759.wav
# Copying file #: 8  153364.wav
# Copying file #: 9  212118.wav
# Copying file #: 10  235480.wav
# Copying file #: 11  121614.wav
# Copying file #: 12  181758.wav
# Copying file #: 13  121608.wav
# Copying file #: 14  423338.wav
# Copying file #: 15  181760.wav
# Copying file #: 16  428815.wav
# Copying file #: 17  223341.wav
# Copying file #: 18  393910.wav
# Copying file #: 19  423344.wav
# Copying file #: 20  276230.wav
# Copying file #: 21  29649.wav
# Copying file #: 22  276170.wav
# Copying file #: 23  276200.wav
# Copying file #: 24  188754.wav
# Copying file #: 25  240780.wav
# Copying file #: 26  324699.wav
# Copying file #: 27  132938.wav
# Copying file #: 28  276176.wav
# Copying file #: 29  276129.wav
# Processing label: Coughing
# Copying file #: 0  81087.wav
# Copying file #: 1  257770.wav
# Copying file #: 2  109640.wav
# Copying file #: 3  45150.wav
# Copying file #: 4  240374.wav
# Copying file #: 5  348364.wav
# Copying file #: 6  323511.wav
# Copying file #: 7  353669.wav
# Copying file #: 8  399619.wav
# Copying file #: 9  31748.wav
# Copying file #: 10  201733.wav
# Copying file #: 11  252231.wav
# Copying file #: 12  187909.wav
# Copying file #: 13  273602.wav
# Copying file #: 14  377267.wav
# Copying file #: 15  331063.wav
# Copying file #: 16  365158.wav
# Copying file #: 17  342669.wav
# Copying file #: 18  325777.wav
# Copying file #: 19  331058.wav
# Copying file #: 20  154434.wav
# Copying file #: 21  19123.wav
# Copying file #: 22  77459.wav
# Copying file #: 23  323695.wav
# Copying file #: 24  211524.wav
# Copying file #: 25  385829.wav
# Copying file #: 26  152995.wav
# Copying file #: 27  426271.wav
# Copying file #: 28  399627.wav
# Copying file #: 29  332989.wav
# Processing label: Opening Drawer
# Copying file #: 0  371300.wav
# Copying file #: 1  211918.wav
# Copying file #: 2  365507.wav
# Copying file #: 3  152310.wav
# Copying file #: 4  334495.wav
# Copying file #: 5  152529.wav
# Copying file #: 6  79387.wav
# Copying file #: 7  183614.wav
# Copying file #: 8  170894.wav
# Copying file #: 9  429144.wav
# Copying file #: 10  66928.wav
# Copying file #: 11  151572.wav
# Copying file #: 12  152530.wav
# Copying file #: 13  152308.wav
# Copying file #: 14  319636.wav
# Copying file #: 15  237698.wav
# Copying file #: 16  245782.wav
# Copying file #: 17  322397.wav
# Copying file #: 18  276440.wav
# Copying file #: 19  151573.wav
# Copying file #: 20  334490.wav
# Copying file #: 21  20086.wav
# Copying file #: 22  152306.wav
# Copying file #: 23  20085.wav
# Copying file #: 24  237379.wav
# Copying file #: 25  319635.wav
# Copying file #: 26  151571.wav
# Copying file #: 27  170893.wav
# Copying file #: 28  211919.wav
# Copying file #: 29  151574.wav
# Processing label: Snapping Fingers
# Copying file #: 0  177496.wav
# Copying file #: 1  169619.wav
# Copying file #: 2  346788.wav
# Copying file #: 3  177500.wav
# Copying file #: 4  361745.wav
# Copying file #: 5  346805.wav
# Copying file #: 6  137148.wav
# Copying file #: 7  425670.wav
# Copying file #: 8  361756.wav
# Copying file #: 9  361769.wav
# Copying file #: 10  88676.wav
# Copying file #: 11  88686.wav
# Copying file #: 12  177497.wav
# Copying file #: 13  361748.wav
# Copying file #: 14  364732.wav
# Copying file #: 15  319811.wav
# Copying file #: 16  361768.wav
# Copying file #: 17  364721.wav
# Copying file #: 18  364720.wav
# Copying file #: 19  361752.wav
# Copying file #: 20  364725.wav
# Copying file #: 21  319812.wav
# Copying file #: 22  257916.wav
# Copying file #: 23  361743.wav
# Copying file #: 24  88677.wav
# Copying file #: 25  88678.wav
# Copying file #: 26  346811.wav
# Copying file #: 27  346813.wav
# Copying file #: 28  361761.wav
# Copying file #: 29  364718.wav
# Processing label: Jangling Keys
# Copying file #: 0  195072.wav
# Copying file #: 1  199731.wav
# Copying file #: 2  274817.wav
# Copying file #: 3  156188.wav
# Copying file #: 4  156180.wav
# Copying file #: 5  336195.wav
# Copying file #: 6  194667.wav
# Copying file #: 7  17874.wav
# Copying file #: 8  324328.wav
# Copying file #: 9  408228.wav
# Copying file #: 10  408224.wav
# Copying file #: 11  156178.wav
# Copying file #: 12  194660.wav
# Copying file #: 13  156187.wav
# Copying file #: 14  176540.wav
# Copying file #: 15  405699.wav
# Copying file #: 16  329049.wav
# Copying file #: 17  119728.wav
# Copying file #: 18  194664.wav
# Copying file #: 19  390368.wav
# Copying file #: 20  405698.wav
# Copying file #: 21  274819.wav
# Copying file #: 22  9231.wav
# Copying file #: 23  416286.wav
# Copying file #: 24  156185.wav
# Copying file #: 25  102960.wav
# Copying file #: 26  85121.wav
# Copying file #: 27  156181.wav
# Copying file #: 28  194666.wav
# Copying file #: 29  256414.wav
# Processing label: Knocking Doors
# Copying file #: 0  218985.wav
# Copying file #: 1  412375.wav
# Copying file #: 2  103995.wav
# Copying file #: 3  103998.wav
# Copying file #: 4  133889.wav
# Copying file #: 5  103999.wav
# Copying file #: 6  364678.wav
# Copying file #: 7  144947.wav
# Copying file #: 8  182024.wav
# Copying file #: 9  148944.wav
# Copying file #: 10  274943.wav
# Copying file #: 11  148947.wav
# Copying file #: 12  364680.wav
# Copying file #: 13  148951.wav
# Copying file #: 14  427613.wav
# Copying file #: 15  399665.wav
# Copying file #: 16  274939.wav
# Copying file #: 17  218981.wav
# Copying file #: 18  427619.wav
# Copying file #: 19  151089.wav
# Copying file #: 20  399663.wav
# Copying file #: 21  182039.wav
# Copying file #: 22  144949.wav
# Copying file #: 23  218987.wav
# Copying file #: 24  144946.wav
# Copying file #: 25  190020.wav
# Copying file #: 26  144943.wav
# Copying file #: 27  412858.wav
# Copying file #: 28  413270.wav
# Copying file #: 29  190025.wav
# Processing label: Laughing
# Copying file #: 0  169803.wav
# Copying file #: 1  46966.wav
# Copying file #: 2  235101.wav
# Copying file #: 3  16203.wav
# Copying file #: 4  344044.wav
# Copying file #: 5  214504.wav
# Copying file #: 6  84606.wav
# Copying file #: 7  351166.wav
# Copying file #: 8  65936.wav
# Copying file #: 9  319365.wav
# Copying file #: 10  344043.wav
# Copying file #: 11  19152.wav
# Copying file #: 12  45129.wav
# Copying file #: 13  196076.wav
# Copying file #: 14  241526.wav
# Copying file #: 15  19164.wav
# Copying file #: 16  235100.wav
# Copying file #: 17  196087.wav
# Copying file #: 18  16204.wav
# Copying file #: 19  150329.wav
# Copying file #: 20  386603.wav
# Copying file #: 21  169801.wav
# Copying file #: 22  39862.wav
# Copying file #: 23  241500.wav
# Copying file #: 24  30282.wav
# Copying file #: 25  213608.wav
# Copying file #: 26  77440.wav
# Copying file #: 27  343942.wav
# Copying file #: 28  196073.wav
# Copying file #: 29  19151.wav
# Processing label: Walking
# Copying file #: 0  422856.wav
# Copying file #: 1  421147.wav
# Copying file #: 2  428506.wav
# Copying file #: 3  348355.wav
# Copying file #: 4  232824.wav
# Copying file #: 5  340217.wav
# Copying file #: 6  202412.wav
# Copying file #: 7  422758.wav
# Copying file #: 8  217947.wav
# Copying file #: 9  92999.wav
# Copying file #: 10  166295.wav
# Copying file #: 11  32635.wav
# Copying file #: 12  153282.wav
# Copying file #: 13  405633.wav
# Copying file #: 14  363874.wav
# Copying file #: 15  166304.wav
# Copying file #: 16  156627.wav
# Copying file #: 17  340224.wav
# Copying file #: 18  72737.wav
# Copying file #: 19  430213.wav
# Copying file #: 20  419447.wav
# Copying file #: 21  321841.wav
# Copying file #: 22  379360.wav
# Copying file #: 23  321848.wav
# Copying file #: 24  153310.wav
# Copying file #: 25  390583.wav
# Copying file #: 26  152788.wav
# Copying file #: 27  186037.wav
# Copying file #: 28  406678.wav
# Copying file #: 29  269856.wav
