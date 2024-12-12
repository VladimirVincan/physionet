create:
	./login.sh
	fmle-cli e create --name "physionet" --docker-image "10.252.13.191:5000/bdsw/vladimir.vincan.ivi/mamba:0.1" --script-entry-point "mamba/train.sh" mamba
update:
	./login.sh
	fmle-cli e update dd1916cc-6faf-4c13-8616-12a26294abaf --docker-image "10.252.13.191:5000/bdsw/vladimir.vincan.ivi/mamba:0.1" --script-entry-point "mamba/train.sh" mamba
update_PointFiveFour:
	./login.sh
	fmle-cli e update dd1916cc-6faf-4c13-8616-12a26294abaf --docker-image "10.252.13.191:5000/bdsw/vladimir.vincan.ivi/mamba:0.1" --script-entry-point "Physionet_PointFiveFour/Training_Code/train.sh" Physionet_PointFiveFour
	# fmle-cli e t launch mamba/train.yaml
