Module subroutines
  USE global
  IMPLICIT NONE

CONTAINS

  Subroutine read_input
    OPEN(UNIT=22,FILE='INPUT',STATUS='old')
    READ(22,NML=indata)
    CLOSE(22)
    
    ax=2*Lx/Nx    
    ay=2*Ly/Ny    
    norm=ax*ay    
    norm_c = sqrt(real(Nx*Ny, dp))/256/sqrt(real(Lx*Ly, dp))*70

    f_p=f_p*(1/norm_c)    

  end Subroutine read_input

  SUBROUTINE  init_pdb
    integer :: ix, iy
    real(dp)  :: sx, sy  
    real(dp) :: re_y1, im_y1, re_y2, im_y2
  
    open(unit=22, file="phcplx-opo_spc"//trim(adjustl(label))//".dat", status='old')    
    read(22, *)
    open(unit=23, file="excplx-opo_spc"//trim(adjustl(label))//".dat", status='old')    
    read(23, *)
    do iy=1, Ny    
       do ix=1, Nx    
          read(22, fmt=' (1x, d12.5, 1x, d12.5, 1x, d12.5, 1x, d12.5) ') sx, sy, re_y1, im_y1
          read(23, fmt=' (1x, d12.5, 1x, d12.5, 1x, d12.5, 1x, d12.5) ') sx, sy, re_y2, im_y2
          pdb(ix,iy,1)= one*re_y1*(1/norm_c) + I*im_y1*(1/norm_c)    
          pdb(ix,iy,2)= one*re_y2*(1/norm_c) + I*im_y2*(1/norm_c)    
       end do    
       read(22,*)    
       read(23,*)    
    end do    
    close(22)    
    close(23)

  END SUBROUTINE  init_pdb

  Subroutine init_pot_c
    integer ix, iy

    ix = int((def_x_pos+Lx)/ax + 1)
    iy = int((def_y_pos+Ly)/ay + 1)
    !for old coordinates of (Nx/2,Ny/2) set def_x_pos = -ax, def_y_pos=-ay

    pot_c = 0
    pot_c(ix,iy)=gv

  end Subroutine init_pot_c

  SUBROUTINE  init_pump_th
    integer :: ix, iy
    real(dp)  :: sx, sy  
		
    !top hat pump    
    open(unit=25, file='pump.dat', status='replace')    
    do iy=1, Ny    
       sy=-Ly+(iy-1)*ay    
       do ix=1, Nx    
          sx=-Lx+(ix-1)*ax    
          pump_spatial(ix,iy)=f_p*0.5*&    
               ( tanh((1.0_dp/10)*( sqrt(sx**2+sy**2)+sigma_p ))-&    
               tanh((1.0_dp/10)*( sqrt(sx**2+sy**2)-sigma_p )) ) + zero    
          write(25,*) sx, sy, abs(pump_spatial(ix,iy))*norm_c    
       end do    
       write(25,*)    
    end do    
    close(25)
        
    do iy=1, Ny    
       !sy=-Ly+(iy-1)*ay    
       do ix=1, Nx    
          sx=-Lx+(ix-1)*ax    
          pump_spatial(ix,iy)= pump_spatial(ix,iy)*cos(k_p*sx)+I*pump_spatial(ix,iy)*sin(k_p*sx)    
       end do    
    end do

  END SUBROUTINE  init_pump_th

  SUBROUTINE  init_pump_homo
    integer :: ix, iy
    real(dp)  :: sx, sy  

    !homogeneous pumping		
    open(unit=25, file='pump.dat', status='replace')    
    do iy=1, Ny    
       sy=-Ly+(iy-1)*ay    
       do ix=1, Nx    
          sx=-Lx+(ix-1)*ax    
          pump_spatial(ix,iy)=f_p + zero    
          write(25,*) sx, sy, abs(pump_spatial(ix,iy))*norm_c    
       end do    
       write(25,*)    
    end do    
    close(25)    
        
    do iy=1, Ny    
       !sy=-Ly+(iy-1)*ay    
       do ix=1, Nx    
          sx=-Lx+(ix-1)*ax    
          pump_spatial(ix,iy)= pump_spatial(ix,iy)*cos(k_p*sx)+I*pump_spatial(ix,iy)*sin(k_p*sx)    
       end do    
    end do

  END SUBROUTINE  init_pump_homo

  Subroutine setg
    integer :: j,k    
        
    DO j=1,(Ny/2+1)    
       DO k=1,(Nx/2+1)    
          kinetic(k,j)=pi**2*(&    
               &(k-1)**2/(Lx**2)+(j-1)**2/(Ly**2))    
       END DO    
    END DO    
    DO j=(Ny/2+2),Ny    
       DO k=(Nx/2+2),Nx    
          kinetic(k,j)=pi**2*( &    
               & (k-1-Nx)**2/(Lx**2)+(j-1-Ny)**2/(Ly**2))    
       END DO    
    END DO    
    DO j=1,(Ny/2+1)    
       DO k=(Nx/2+2),Nx    
          kinetic(k,j)=pi**2*(&    
               &(k-1-Nx)**2/(Lx**2)+(j-1)**2/(Ly**2))    
       END DO    
    END DO    
    DO j=(Ny/2+2),Ny    
       DO k=1,(Nx/2+1)      
          kinetic(k,j)=pi**2*(&    
               &(k-1)**2/(Lx**2)+(j-1-Ny)**2/(Ly**2))    
       END DO    
    END DO    

  end subroutine setg

  Subroutine export_evolution
    integer :: i_t    
    integer :: kx, ky    
  
    real(dp) :: omega    
    real(dp) :: mom_x, mom_y    
  
    !export y_tot_0 to file!!    
    open(unit=28, file="spectr_om-vs-k_no-trigg.dat", status='replace')    
    write(28, fmt=' ("#", 1x, "mom_x", 12x, "mom_y", 12x, "omega", 12x, "real(y_tot_0)", 1x, "aimag(y_tot_0)") ')         
  
    do i_t=Nt/2+2, Nt    
       omega=2*pi*(i_t-1-Nt)/( (Nt-1)*dxsav_sp )    
       do ky=Ny/2+2, Ny  
          mom_y=pi*(ky-1-Ny)/Ly    
          do kx=Nx/2+2, Nx    
             mom_x=pi*(kx-1-Nx)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
          do kx=1, Nx/2+1    
             mom_x=pi*(kx-1)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
       end do    
       do ky=1, Ny/2+1    
          mom_y=pi*(ky-1)/Ly    
          do kx=Nx/2+2, Nx    
             mom_x=pi*(kx-1-Nx)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
          do kx=1, Nx/2+1    
             mom_x=pi*(kx-1)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
    end do    
    end do    
    do i_t=1, Nt/2+1    
       omega=2*pi*(i_t-1)/( (Nt-1)*dxsav_sp )    
       do ky=Ny/2+2, Ny  
          mom_y=pi*(ky-1-Ny)/Ly    
          do kx=Nx/2+2, Nx    
             mom_x=pi*(kx-1-Nx)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
          do kx=1, Nx/2+1    
             mom_x=pi*(kx-1)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
       end do    
       do ky=1, Ny/2+1    
          mom_y=pi*(ky-1)/Ly    
          do kx=Nx/2+2, Nx    
             mom_x=pi*(kx-1-Nx)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
          do kx=1, Nx/2+1    
             mom_x=pi*(kx-1)/Lx    
             write(28,100) mom_x, mom_y, -omega, real(y_tot_0(kx,ky,i_t)), aimag(y_tot_0(kx,ky,i_t))    
          end do    
    end do    
    end do    
    close(28)                

    100 format (5(1x, d12.5))
  end Subroutine export_evolution    

  Subroutine import_evolution
    integer :: i_t    
    integer :: kx, ky    
  
    real(dp) :: re_y_tot_0, im_y_tot_0    
    real(dp) :: omega    
    real(dp) :: mom_x, mom_y
  
    y_tot_0=zero    
  
    !!importing y_tot_0!!    
    open(unit=28, file="spectr_om-vs-k_no-trigg.dat", status='old')    
    read(28, fmt=' ("#", 1x, "mom_x", 12x, "mom_y", 12x, "omega", 12x, "real(y_tot_0)", 1x, "aimag(y_tot_0)") ')         

    do i_t=Nt, 1, -1    
       do ky=Ny/2+2, Ny  
          do kx=Nx/2+2, Nx    
             read(28,101) mom_x, mom_y, omega, re_y_tot_0, im_y_tot_0    
             y_tot_0(kx,ky,i_t)=one*re_y_tot_0+I*im_y_tot_0    
          end do    
          do kx=1, Nx/2+1    
             read(28,101) mom_x, mom_y, omega, re_y_tot_0, im_y_tot_0    
             y_tot_0(kx,ky,i_t)=one*re_y_tot_0+I*im_y_tot_0    
          end do    
       end do    
       do ky=1, Ny/2+1    
          do kx=Nx/2+2, Nx    
             read(28,101) mom_x, mom_y, omega, re_y_tot_0, im_y_tot_0    
             y_tot_0(kx,ky,i_t)=one*re_y_tot_0+I*im_y_tot_0    
          end do    
          do kx=1, Nx/2+1    
             read(28,101) mom_x, mom_y, omega, re_y_tot_0, im_y_tot_0    
             y_tot_0(kx,ky,i_t)=one*re_y_tot_0+I*im_y_tot_0    
          end do    
       end do    
    end do    
       
    close(28)

    101 format (5(1x, d12.5))
  end Subroutine import_evolution

  Subroutine eval_spectr_0
    integer :: i_t    
    integer :: kx
    real(dp) :: omega    
    real(dp) :: mom_x
    
    !full spectum	
    open(unit=23, file="spectr_om-vs-kx_no-trigg.dat", status='replace')    
    write(23, fmt=' ("#", 1x, "mom_x", 19x, "omega", 19x, "abs(psi(1))**2") ')    
    do i_t=Nt/2+2, Nt    
       omega=2*pi*(i_t-1-Nt)/( (Nt-1)*dxsav_sp )    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(23, *) mom_x, -omega, abs(y_tot_0(kx,1,i_t))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(23, *) mom_x, -omega, abs(y_tot_0(kx,1,i_t))**2    
       end do    
       write(23,*)    
    end do    
    do i_t=1, Nt/2+1    
       omega=2*pi*(i_t-1)/( (Nt-1)*dxsav_sp )    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(23, *) mom_x, -omega, abs(y_tot_0(kx,1,i_t))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(23, *) mom_x, -omega, abs(y_tot_0(kx,1,i_t))**2    
       end do    
       write(23,*)    
    end do    
    close(23)    
  
    !integrated spectrum	
    open(unit=24, file="int-spectr_no-trigg.dat", status='replace')    
    write(24, fmt=' ("#", 1x, "omega", 20x, "int_omega") ')    
    int_sp=sum(sum(abs(y_tot_0), dim=1), dim=1)    
    do i_t=Nt/2+2, Nt    
       omega=2*pi*(i_t-1-Nt)/( (Nt-1)*dxsav_sp )    
       write(24, *) -omega, int_sp(i_t)    
    end do    
    do i_t=1, Nt/2+1    
       omega=2*pi*(i_t-1)/( (Nt-1)*dxsav_sp )    
       write(24, *) -omega, int_sp(i_t)    
    end do    
    close(24)    
                
  end Subroutine eval_spectr_0    

  Subroutine filter_peak(omega, omega_cut, lbl)
    real(dp), intent (in) :: omega, omega_cut
    character(len=*), intent (in) :: lbl

    integer :: i_t, num
    integer :: i_tmax, i_tmax_i, i_tmax_f

    real(dp) :: omega_i, omega_f

    write(*,*) trim(adjustl(lbl))
    write(*,*) 'omega= ', omega

    !find index of peak
    i_tmax = int(1 + Nt/2 + omega/(2*pi) * (Nt-1)*dxsav_sp)
    if ((i_tmax.lt.1).or.(i_tmax.gt.Nt)) then
        write(*,*) "index of "//trim(adjustl(lbl))//" peak out of range!"
    end if

    !calculating indices of energy window
    omega_i = omega - omega_cut
    write(*,*) 'omega_i= ', omega_i
    i_tmax_i = int(1 + Nt/2 + omega_i/(2*pi) * (Nt-1)*dxsav_sp)
    if ((i_tmax_i.lt.1).or.(i_tmax_i.gt.Nt)) then
        write(*,*) "L index of "//trim(adjustl(lbl))//" out of range!"
    end if

    omega_f = omega + omega_cut
    write(*,*) 'omega_f= ', omega_f
    i_tmax_f = int(1 + Nt/2 + omega_f/(2*pi) * (Nt-1)*dxsav_sp)
    if ((i_tmax_f.lt.1).or.(i_tmax_f.gt.Nt)) then
        write(*,*) "R index of "//trim(adjustl(lbl))//" out of range!"
    end if
		
    !integrate in energy
    write(*,*) 'i_tmax_i= ', i_tmax_i    
    write(*,*) 'i_tmax_f= ', i_tmax_f    
    write(*,*)    
        
    y_enfilt=zero
    
    if ((i_tmax_f-i_tmax_i).eq.1) i_tmax_i=i_tmax_f

    do i_t=i_tmax_i, i_tmax_f
       y_enfilt(:,:)=y_enfilt(:,:) + y_tot_0(:,:,i_t)
    end do

    !normalize
    num = i_tmax_f - i_tmax_i + 1
    y_enfilt(:,:)=y_enfilt(:,:)/num

  end Subroutine filter_peak

  Subroutine write_kx_max(lbl) 
    character(len=*), intent (in) :: lbl
    
    real(dp) :: mom_x_max
    
    !write kx of maximum peak emission
    kx_max=maxloc( abs(y_enfilt(:,1)) )    
    if ( kx_max(1).ge.Nx/2+2 ) mom_x_max=pi*(kx_max(1)-1-Nx)/Lx    
    if ( kx_max(1).le.Nx/2+1 ) mom_x_max=pi*(kx_max(1)-1)/Lx    
    write(25,*) "mom_x_max_"//trim(adjustl(lbl))//"= ", mom_x_max    

  end Subroutine write_kx_max

  Subroutine write_peak_mom(lbl)
    character(len=*), intent (in) :: lbl
    
    integer :: kx, ky
    real(dp) :: mom_x, mom_y
  
    !write peak emission in momentum
    open(unit=27, file="opo_ph-mom_enfilt_"//trim(adjustl(lbl))//".dat", status='replace')    
    write(27, fmt=' ("#", 1x, "kx", 12x, "ky", 12x, "|psi(1)|^2") ')    
    do ky=Ny/2+2, Ny    
       mom_y=pi*(ky-1-Ny)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(27,*) mom_x, mom_y, &    
               abs(y_enfilt(kx,ky))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(27,*) mom_x, mom_y, &    
               abs(y_enfilt(kx,ky))**2    
       end do    
       write(27,*)    
    end do    
    do ky=1, Ny/2+1    
       mom_y=pi*(ky-1)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(27,*) mom_x, mom_y, &    
               abs(y_enfilt(kx,ky))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(27,*) mom_x, mom_y, &    
               abs(y_enfilt(kx,ky))**2    
       end do    
       write(27,*)    
    end do    
    close(27)    

  end Subroutine write_peak_mom
  
  Subroutine write_peak_spc(lbl)
    character(len=*), intent (in) :: lbl
    
    integer :: ix, iy
    real(dp) :: sx, sy
	
    !write peak emission in space    
    open(unit=26, file="opo_ph-spc_enfilt_"//trim(adjustl(lbl))//".dat", status='replace')    
    write(26, fmt=' ("#", 1x, "x", 12x, "y", 12x, "|psi(1)|^2") ')    
    do iy=1, Ny    
       sy=-Ly+(iy-1)*ay    
       do ix=1, Nx    
          sx=-Lx+(ix-1)*ax    
          write(26,*) sx, sy, abs(y_enfilt(ix,iy))**2    
       end do    
       write(26,*)    
    end do    
    close(26)    

  end Subroutine write_peak_spc
  
  Subroutine mom_filter
    integer :: kx, ky    
    real(dp) :: mom_x, mom_y    
  
    wave_f_flt=zero    
    do ky=Ny/2+2, Ny    
       mom_y=pi*(ky-1-Ny)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          if( sqrt((mom_x-mom_cent)**2+mom_y**2) .le. mom_cut ) then    
             wave_f_flt(kx,ky,1) = wave_f_mom(kx,ky,1)    
          end if    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          if( sqrt((mom_x-mom_cent)**2+mom_y**2) .le. mom_cut ) then    
             wave_f_flt(kx,ky,1) = wave_f_mom(kx,ky,1)    
          end if    
       end do    
    end do    
    do ky=1, Ny/2+1    
       mom_y=pi*(ky-1)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          if( sqrt((mom_x-mom_cent)**2+mom_y**2) .le. mom_cut ) then    
             wave_f_flt(kx,ky,1) = wave_f_mom(kx,ky,1)    
          end if    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          if( sqrt((mom_x-mom_cent)**2+mom_y**2) .le. mom_cut ) then    
             wave_f_flt(kx,ky,1) = wave_f_mom(kx,ky,1)    
          end if    
       end do    
    end do    

  end Subroutine mom_filter    

  Subroutine write_momentum
    integer :: kx, ky    
    real(dp) :: mom_x, mom_y    
  
    open(unit=24, file="opo_mom_ph"//trim(adjustl(label))//".dat", status='replace')    
    write(24, fmt=' ("#", 1x, "kx", 12x, "ky", 12x, "|psi(1)|^2") ')    
    do ky=Ny/2+2, Ny    
       mom_y=pi*(ky-1-Ny)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(24,*) mom_x, mom_y,&    
               abs(wave_f_mom(kx,ky,1))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(24,*) mom_x, mom_y,&    
               abs(wave_f_mom(kx,ky,1))**2    
       end do    
       write(24,*)    
    end do    
    do ky=1, Ny/2+1    
       mom_y=pi*(ky-1)/Ly    
       do kx=Nx/2+2, Nx    
          mom_x=pi*(kx-1-Nx)/Lx    
          write(24,*) mom_x, mom_y,&    
               abs(wave_f_mom(kx,ky,1))**2    
       end do    
       do kx=1, Nx/2+1    
          mom_x=pi*(kx-1)/Lx    
          write(24,*) mom_x, mom_y,&    
               abs(wave_f_mom(kx,ky,1))**2    
       end do    
       write(24,*)    
    end do    
    close(24)    
  
	!write cut in momentum
    open(unit=25, file="cutky0opo_mom_ph"//trim(adjustl(label))//".dat", status='replace')    
    write(25, fmt=' ("#", 1x, "kx", 12x, "ky", 12x, "|psi(1)|^2") ')    
    ky=1    
    mom_y=pi*(ky-1)/Ly    
    do kx=Nx/2+2, Nx    
       mom_x=pi*(kx-1-Nx)/Lx    
       write(25,*) mom_x, mom_y,&    
            abs(wave_f_mom(kx,ky,1))**2    
    end do    
    do kx=1, Nx/2+1    
       mom_x=pi*(kx-1)/Lx    
       write(25,*) mom_x, mom_y,&    
            abs(wave_f_mom(kx,ky,1))**2    
    end do    
    close(25)    

  end Subroutine write_momentum    

end Module subroutines
