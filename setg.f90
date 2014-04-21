subroutine setg(Nx, Ny, Lx, Ly, kinetic)
  implicit none

  integer :: Nx, Ny
  real :: Lx, Ly
  real :: kinetic(Nx, Ny)

  integer :: j,k    
  real, parameter :: pi=3.141592653589793238462643383279502884197

  !f2py integer, intent(in) :: Nx, Ny
  !f2py real, intent(in) :: Lx, Ly
  !f2py real, intent(out), dimension(Nx, Ny) :: kinetic

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
