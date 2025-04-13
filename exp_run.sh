
#!/bin/bash
# Generate random test data
python main.py --eqn_type=pricing_default_risk --cur_dim=10 --model_type=heap --drop_out=1 --with_label=0 --train_mode=0

# Train and test using CSO and Heap
python main.py --eqn_type=pricing_default_risk --cur_dim=10 --model_type=cso --drop_out=1 --with_label=0 --train_mode=1 --test_num=1
python main.py --eqn_type=pricing_default_risk --cur_dim=10 --model_type=heap --drop_out=1 --with_label=0 --train_mode=1 --test_num=1