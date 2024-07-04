create:
	./login.sh
	fmle-cli e create --name "physionet" --docker-image "10.252.13.191:5000/bdsw/vladimir.vincan.ivi/mamba:0.1" --script-entry-point "mamba/train.sh" mamba
update:
	./login.sh
	fmle-cli e update "physionet" --docker-image "10.252.13.191:5000/bdsw/vladimir.vincan.ivi/mamba:0.1" --script-entry-point "mamba/train.sh" mamba
	# fmle-cli e t launch mamba/train.yaml
