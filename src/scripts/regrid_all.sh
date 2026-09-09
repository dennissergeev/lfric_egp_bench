pixi run regrid -e shj_c48_l32 --nlat 96 --nlon 192 --overwrite
pixi run regrid -e shj_c24_l32 --nlat 48 --nlon 96 --overwrite
pixi run regrid -e shj_c96_l32 --nlat 192 --nlon 384 --overwrite
pixi run regrid -e shj_c48_l16 --nlat 96 --nlon 192 --overwrite
pixi run regrid -e shj_c48_l64 --nlat 96 --nlon 192 --overwrite

pixi run regrid -e dhj_c48_l66 --nlat 96 --nlon 192 --overwrite
pixi run regrid -e dhj_c24_l66 --nlat 48 --nlon 96 --overwrite
pixi run regrid -e dhj_c96_l66 --nlat 192 --nlon 384 --overwrite
pixi run regrid -e dhj_c48_l66_s0p5_m90 --nlat 192 --nlon 384 --overwrite
pixi run regrid -e dhj_c48_l66_s0p5_p90 --nlat 192 --nlon 384 --overwrite