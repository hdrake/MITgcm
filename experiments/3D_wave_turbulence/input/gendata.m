clear

%--------
nx = 100;
ny = 100;
nz = 350;

dx = 30;
dy = 30;
dz = 10;

dz = dz * ones(nz, 1);
dz(251:350) = dz(250) * 1.02.^(1:100);
% --------

%--------
Lx = nx*dx;
Ly = ny*dy;
Hz = sum(dz);

x  = (dx/2:dx:Lx-dx/2);
y  = (dy/2:dy:Ly-dy/2);
z  = - Hz + cumsum(dz) - dz/2;

%

[X, Y, Z] = meshgrid(x, y, z);
%--------

%--------
load hbot.mat

%---- Henri Drake, edited 09/18/2018 ----
% subsample Hbot for 3km section
Hbotnew = zeros(ny,nx);
for j=1:ny; Hbotnew(j,:) = Hbot(1:nx); end
Hbot = Hbotnew;

%figure(1)
%subplot(1,2,1)
%pcolor(x,y,Hbot)
%subplot(1,2,2)
%plot(x,Hbot(1,:));

[xx, yy] = meshgrid(x, y);
Hbot = Hbot+(1-xx/xx(1,nx))*(Hbot(1,nx)-Hbot(1,1));
%----------------------------------------
Hbot = Hbot - min(Hbot(:));
Hbot = Hbot - Hz;

%figure(2)
%subplot(1,2,1)
%pcolor(x,y,Hbot)
%subplot(1,2,2)
%plot(x,Hbot(1,:));
% --------

%--------
U = zeros(nx, ny, nz);
V = 0.025 * 0.53e-4/1.4e-4 * ones(nx, ny, nz);
T = (1e-3)^2 *(Z+Hz) /9.81/2e-4;

% Add perturbations to break meridional homogeneity
Tperturb = 0.001*rand([ny,1])';
for k=1:nz
    for i=1:nx
        T(i,:,k) = T(i,:,k) + Tperturb;
    end
end

R = squeeze(T(1, 1, :));
%--------

%%

% ---- Henri Drake 09/10/2018 ---------
% Initial tracer concentration
C = exp(-((Y-1000)/200).^2-((Z+5600)/100).^2);

%-----------------------
% Flip vertical direction and swap dimensions to be [nx,ny,nz]
U = flip(U,3);
V = flip(V,3);
T = flip(T,3);
R = flip(R,1);
C = flip(C,3);
dz = flip(dz,1);

fid = fopen('U.init', 'w', 'b'); 
fwrite(fid, U, 'real*8'); 
fclose(fid);

fid = fopen('V.init', 'w', 'b'); 
fwrite(fid, V, 'real*8'); 
fclose(fid);

fid = fopen('T.init', 'w', 'b'); 
fwrite(fid, T, 'real*8'); 
fclose(fid);

save('R.init', 'R', '-ascii');

fid = fopen('ptracer_init.bin', 'w', 'b');
fwrite(fid, C, 'real*8');
fclose(fid);

fid = fopen('topog.init', 'w', 'b'); 
fwrite(fid, Hbot', 'real*8'); 
fclose(fid);

fid=fopen('delZ.init', 'w', 'b'); 
fwrite(fid, dz, 'real*8'); 
fclose(fid);


