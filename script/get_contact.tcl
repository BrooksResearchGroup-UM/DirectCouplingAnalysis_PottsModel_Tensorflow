# get_contact.tcl

# Purpose:
# Created by Xinqiang Ding (xqding@umich.edu)
# at 2017/05/08 17:47:51
##################################################

## proc for calculate the minimum distance between two atomselections
proc min_dist {sel1 sel2} {
    set min_d  inf
    foreach coor1 [$sel1 get {x y z}] {
	foreach coor2 [$sel2 get {x y z}] {
	    set tmp [vecdist $coor1 $coor2]
	    if {$tmp < $min_d} {set min_d $tmp}
	}
    }
    return $min_d
}

## load pdb file
mol new ./proteins_MSA/1TEN.pdb

## get resid
set sel [atomselect top "protein"]
set resid_list [lsort -unique -integer [$sel get resid]]

## calc min_dist for each pair of resids
## set outfile [open "./output/Ras/dist_pdb.txt" w]
set outfile [open "./output/dist_pdb.txt" w]

foreach i $resid_list {
    foreach j $resid_list {
	if {$j > $i} {
	    set sel1 [atomselect top "protein and  resid $i"]
	    set sel2 [atomselect top "protein and  resid $j"]
	    puts $outfile $i,$j,[min_dist $sel1 $sel2]
	}
    }
}
