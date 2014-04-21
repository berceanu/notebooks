subroutine init_pump_th(Nx, Ny, Lx, Ly, ax, ay, f_p, sigma_p, k_p, pump_spatial)
  implicit none
  integer :: ix, iy, Nx, Ny
  real :: sx, sy, Lx, Ly, ax, ay, f_p, sigma_p, k_p
  complex :: pump_spatial(Nx, Ny)
  complex, parameter :: I = (0.0, 1.0)    ! complex i
  complex, parameter :: zero = (0.0, 0.0) ! complex 0
      	
  !f2py integer, intent(in) :: Nx, Ny
  !f2py real, intent(in) :: Lx, Ly, ax, ay, f_p, sigma_p, k_p
  !f2py complex, intent(out), dimension(Nx, Ny) :: pump_spatial

  !top hat pump    
  do iy=1, Ny    
     sy=-Ly+(iy-1)*ay    
     do ix=1, Nx    
        sx=-Lx+(ix-1)*ax    
        pump_spatial(ix,iy)=f_p*0.5*&    
             ( tanh((1.0/10)*( sqrt(sx**2+sy**2)+sigma_p ))-&    
             tanh((1.0/10)*( sqrt(sx**2+sy**2)-sigma_p )) ) + zero    
     end do    
  end do    
      
  do iy=1, Ny    
     do ix=1, Nx    
        sx=-Lx+(ix-1)*ax    
        pump_spatial(ix,iy)= pump_spatial(ix,iy)*cos(k_p*sx)+I*pump_spatial(ix,iy)*sin(k_p*sx)    
     end do    
  end do

end subroutine init_pump_th
