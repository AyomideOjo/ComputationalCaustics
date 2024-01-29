%quick test for doing voronoi and Lloyd's relaxation in matlab
clear 
clf

imdata = imread("mlk_square.jpg");

Adata = imdata(:,:,3); % red channel (arbitrarily)
subplot(1,2,1);
imp = imshow(Adata);
subplot(1,2,2);

[n,m] = size(Adata); % n rows, c cols, i.e., y size then x size
[x,y] = meshgrid(1:m,1:n); % meshgrid takes x,y size

gp = [ reshape(x,[],1), reshape(y,[],1) ]; % pixel locations (grid points)
% swap white and black, i.e., high weight for black points and make sure we
% avoid division by zero by staying 1e-3 above 0
A = 1.001 - double(reshape(Adata,[],1))/255;

N = 9000;

% Need to seed these points near the dark (or bright) parts of the image
% otherwise it will take forever to converge
seedp = gp(A > 0.5,:);
ix = randperm(size(seedp,1),N);
p = seedp( ix, : );

% Find ID for each pixel, relatively quickly with NN search
dt = delaunayTriangulation(p);
ID = nearestNeighbor( dt, gp );
vr = imagesc(reshape(ID,n,m));

vr.AlphaData = 0.2;
hold on;
ph0 = plot( p(:,1), p(:,2), '.', 'MarkerSize', 8, 'Color', [0,0,0]);
axis equal
axis tight

maxiter = 30;
for it = 1:maxiter
    disp(it)
    cw = zeros(N,2);
    for i = 1:N
        % compute image data weighted COM for each region (slow?)
        cw(i,:) = mean(A(ID==i).*gp(ID==i,:)) / mean(A(ID==i));
    end
    p = cw; % move the points to the weighted average location
    dt = delaunayTriangulation(p);
    ID = nearestNeighbor( dt, gp );
    vr.CData = reshape(ID, n,m);
    ph0.XData = p(:,1);
    ph0.YData = p(:,2);
    f = getframe;
end