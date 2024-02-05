set_save_dir() {
    mdir=$1
    if [[ -d $mdir ]]; then
        save_dir=${mdir}/eval/$2
    else
        save_dir=$n/space10/out/$(basename $mdir)/eval/$2
    fi
}

set_valid_dir() {
    mdir=$1
    if [[ -d $mdir ]]; then
        save_dir=${mdir}/valid/$2
    else
        save_dir=$n/space10/out/$(basename $mdir)/valid/$2
    fi
}


export set_save_dir
export set_valid_dir

