#! /bin/bash

cd webnlg-automatic-evaluation/
python3 make_eval_files.py
readarray -t teams < teams_list.txt
echo "${teams[@]}"

# BLEU
for TEAMR in $teams
do 
    echo $TEAMR

    . bleu_eval_3ref.sh
    cd ..
    echo "ALL:"; cat webnlg-automatic-evaluation/eval/bleu3ref-$TEAMR\_all-cat.txt > results/$TEAMR/bleu_all.txt
    # BLEU seen
    echo "SEEN:"; cat webnlg-automatic-evaluation/eval/bleu3ref-$TEAMR\_old-cat.txt > results/$TEAMR/bleu_seen.txt
    # BLEU unseen
    echo "UNSEEN:"; cat webnlg-automatic-evaluation/eval/bleu3ref-$TEAMR\_new-cat.txt > results/$TEAMR/bleu_unseen.txt

    # METEOR
    cd meteor-1.5/ 
    ../webnlg-automatic-evaluation/meteor_eval.sh 

    cd ..
    echo "ALL:"; cat webnlg-automatic-evaluation/eval/meteor-$TEAMR-all-cat.txt > results/$TEAMR/meteor_all.txt
    # METEOR seen
    echo "SEEN:"; cat webnlg-automatic-evaluation/eval/meteor-$TEAMR-old-cat.txt > results/$TEAMR/meteor_seen.txt
    # METEOR unseen
    echo "UNSEEN:"; cat webnlg-automatic-evaluation/eval/meteor-$TEAMR-new-cat.txt > results/$TEAMR/meteor_unseen.txt

    # TER
    cd tercom-0.7.25/
    ../webnlg-automatic-evaluation/ter_eval.sh 
    cd ..
    echo "ALL:"; cat webnlg-automatic-evaluation/eval/ter3ref-$TEAMR-all-cat.txt > results/$TEAMR/ter_all.txt
    # TER seen
    echo "SEEN:"; cat webnlg-automatic-evaluation/eval/ter3ref-$TEAMR-old-cat.txt > results/$TEAMR/ter_seen.txt
    # TER unseen
    echo "UNSEEN:"; cat webnlg-automatic-evaluation/eval/ter3ref-$TEAMR-new-cat.txt > results/$TEAMR/ter_unseen.txt

    python3 print_scores_webnlg.py $TEAMR
done
