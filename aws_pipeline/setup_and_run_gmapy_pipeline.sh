user_remote_awareness="$1"
branch_name="$2"

if [ "$user_remote_awareness" != "remote" ]; then
    echo "This script is intended to be run remotely on an AWS machine." >&2
    echo "Add 'remote' as first command line argument if you know " \
         "what you are doing." >&2
    exit 1
fi

# inline script that will be saved to a file
# and called. Checks if results are available
# and then copy them to s3 storage and terminate instance.
export watchguard_script=$(cat <<'EOF'
#!/bin/bash
branch_name="$1"
while [ ! -e "finished_sampling" ]; do
    sleep 10
done

aws s3 cp "." "s3://gmapy-results/$branch_name/" --recursive

# terminate instance if copying successful
if [ "$?" -eq 0 ]; then
    metadata_url=http://169.254.169.254/latest/meta-data
    region=$(curl -s $metadata_url/placement/availability-zone | sed -e 's/.$//')
    instance_id=$(curl -s $metadata_url/instance-id)
    aws ec2 terminate-instances --instance-ids $instance_id --region $region
fi
EOF
)

# disable interactive popups
# see https://stackoverflow.com/questions/73397110/how-to-stop-ubuntu-pop-up-daemons-using-outdated-libraries-when-using-apt-to-i
sudo sed -i "/#\$nrconf{restart} = 'i';/s/.*/\$nrconf{restart} = 'a';/" \
         /etc/needrestart/needrestart.conf
sudo apt update
sudo apt install -y build-essential
sudo apt install -y awscli
sudo apt install -y libsuitesparse-dev
sudo apt install -y python-is-python3
sudo apt install -y python3-pip
sudo apt install -y python3.10-venv
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev

# prepare the gmapy package (including the examples)
git clone --recurse-submodules https://github.com/iaea-nds/neutron-standards-evaluation && \
cd neutron-standards-evaluation && \
git checkout $branch_name && \
git submodule update --init --recursive

if [ $? -ne 0 ]; then
  exit 1
fi

python -m venv calcvenv
source calcvenv/bin/activate
python -m pip install ./gmapy
deactivate

echo -e "$watchguard_script" > "evaluation/watchguard_script.sh"

# run calculation pipeline
echo "---- Starting the gmapy calculation pipeline ----"
screen -S gmapy_session -d -m
screen -S gmapy_session -X stuff "source calcvenv/bin/activate &&"
screen -S gmapy_session -X stuff "cd \"evaluation\" &&"
screen -S gmapy_session -X stuff "python 01_model_preparation.py && touch finished_preparation &&"
screen -S gmapy_session -X stuff "python 02_parameter_optimization.py && touch finished_optimization &&"
screen -S gmapy_session -X stuff "python 03_mcmc_sampling.py && touch finished_sampling &&"
screen -S gmapy_session -X stuff "bash watchguard_script.sh \"$branch_name\" $(printf \\r)"
screen -S gmapy_session -X detach
