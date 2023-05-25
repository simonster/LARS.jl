using LARS, Test, DelimitedFiles, Statistics, LinearAlgebra
using StableRNGs

# Diabetes data set from Efron, Hastie, Johnstone, and Tibshirani (2004)
diabeetus = readdlm(joinpath(dirname(@__FILE__), "diabeetus.csv"), ',', header=true)[1]
y = diabeetus[:, 2]
X = diabeetus[:, 3:end]

# From scikit-learn
lambdas = [949.4352603840745,889.315990734972,452.9009689082005,316.0740526984122,130.13085130152092,88.78242981548489,68.9652212024215,19.981254678104257,5.477472946045879,5.089178805603155,2.1822497288276317,1.3104352485144606,0.0]
coefs = [
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -5.718948001208728 -7.011245148933486 -10.012197817492563
 0.0 0.0 0.0 0.0 0.0 -74.9165139419965 -111.97855445747084 -197.75650113516116 -226.13366183284722 -227.17579824290468 -234.3976216440481 -237.10078599953886 -239.819089365671
 0.0 60.11926964909101 361.8946124552187 434.7579596172354 505.6595584740103 511.3480706973669 512.0440889861112 522.2648470180375 526.885466712335 526.3905943499628 522.6487857596884 521.0751302031779 519.839786790058
 0.0 0.0 0.0 79.2364468833807 191.26988357571105 234.154616158865 252.52701650167552 297.15973688913124 314.38927158204285 314.9504672168165 320.34255435496925 321.54902678167764 324.3904276894301
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -103.94624876685306 -195.1058295076527 -237.34097311836078 -554.2663277460415 -580.4386001513628 -792.1841616279723
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 33.628274414734115 286.73616838072263 313.86213163656953 476.7458378233891
 0.0 0.0 0.0 0.0 -114.10097989037126 -169.71139350739367 -196.04544328664682 -223.92603333538784 -152.47725948507087 -134.59935205248814 0.0 0.0 101.04457032117308
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 106.34280588176777 111.38412869339244 148.9004446350953 139.857867665755 177.0641762322911
 0.0 0.0 301.77534280616015 374.91583685203716 439.66494175551986 450.66744820766553 452.3927277147172 514.7494808474937 529.9160306597569 545.4825972125902 663.0332872920303 674.9366167828011 751.2793210872194
 0.0 0.0 0.0 0.0 0.0 0.0 12.078152260966887 54.76768063022275 64.48741790214113 64.60667013141351 66.33095501209934 67.17939964121148 67.62538639102792
]

c = lars(X, y)
@test c.lambdas ≈ lambdas
@test c.coefs ≈ coefs

# Covariance test results will differ slightly with estimated variance,
# but should match R covTest without estimate
drop_in_cov = [57079.3544,148636.4184,16309.9614,17602.5131,14269.1173,457.8277,9624.1289,1786.1439,407.7183,47.0376]
t = covtest(c, X, y, errorvar=1.0)
@test t.drop_in_cov[1:end-1] ≈ drop_in_cov atol=1e-4

# With penalty
c1 = lars(X, y, lambda2=0.02)
c2 = lars([X .- mean(X, dims=1); sqrt(0.02)*Matrix(I,size(X, 2),size(X, 2))], [y .- mean(y); zeros(size(X, 2))], standardize=false, intercept=false)
@test c1.coefs' ≈ c2.coefs'
c1 = lars(X, y, lambda2=0.02, use_gram=false)
c2 = lars([X .- mean(X, dims=1); sqrt(0.02)*Matrix(I,size(X, 2),size(X, 2))], [y .- mean(y); zeros(size(X, 2))], standardize=false, intercept=false, use_gram=false)
@test c1.coefs' ≈ c2.coefs'

lambdas = [949.4352603840745,889.315990734972,452.9009689082005,316.0740526984122,130.13085130152092,88.78242981548489,68.9652212024215,19.981254678104257,5.477472946045879,5.089178805603155,0.0]
coefs = [
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -10.012197817492368
 0.0 0.0 0.0 0.0 0.0 -74.9165139419965 -111.97855445747084 -197.75650113516116 -226.13366183284722 -227.17579824290468 -239.81908936567078
 0.0 60.11926964909101 361.8946124552187 434.7579596172354 505.6595584740103 511.3480706973669 512.0440889861112 522.2648470180375 526.885466712335 526.3905943499628 519.8397867900582
 0.0 0.0 0.0 79.2364468833807 191.26988357571105 234.154616158865 252.52701650167552 297.15973688913124 314.38927158204285 314.9504672168165 324.3904276894299
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -103.94624876685306 -195.1058295076527 -237.34097311836078 -792.1841616279617
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 33.628274414734115 476.7458378233804
 0.0 0.0 0.0 0.0 -114.10097989037126 -169.71139350739367 -196.04544328664682 -223.92603333538784 -152.47725948507087 -134.59935205248814 101.04457032116852
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 106.34280588176777 111.38412869339244 177.06417623228987
 0.0 0.0 301.77534280616015 374.91583685203716 439.66494175551986 450.66744820766553 452.3927277147172 514.7494808474937 529.9160306597569 545.4825972125902 751.2793210872154
 0.0 0.0 0.0 0.0 0.0 0.0 12.078152260966887 54.76768063022275 64.48741790214113 64.60667013141351 67.62538639102786
]

c = lars(X, y, method=:lar)
@test c.lambdas ≈ lambdas
@test c.coefs ≈ coefs

# Randomly generated data with predictor drops
testdata = readdlm(joinpath(dirname(@__FILE__), "testdata.csv"), ',')
y = testdata[:, 1]
X = testdata[:, 2:end]

# Unstandardized from scikit-learn
lambdas = [766.1750193137241,542.1829897748706,523.136666260976,374.9244286861718,264.8489755413771,21.96519890364044,6.925037097241922,5.959772604690446,1.923172335288859,0.0]
coefs = [
 0.0 0.0 0.03475908385654046 0.22208390493742114 0.7186532728887909 1.4295258110986508 1.7608698681512005 1.8842516592288858 1.9443119146326815 2.646040188752718
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.12044402668354513 0.1074715996305585 0.8640940999704245
 0.0 0.0 0.0 0.23772230110830994 2.7755575615628914e-17 0.0 -0.3916472749621872 -0.5007864804329035 -0.5786646701886231 -1.1746378982372363
 0.0 -0.3621673838611561 -0.38689480994549247 -0.5515575656651571 -0.4382857790715998 -0.39607662702296464 -0.2383632019845157 -2.7755575615628914e-17 0.0 1.4717111134984753
 0.0 0.0 0.0 0.0 -0.46692708190385723 -1.0548894711735968 -1.42155776999432 -1.5787904092773575 -1.640778357039719 -2.5495821284311173
]

c = lars(X, y, standardize=false, intercept=false)
@test c.lambdas ≈ lambdas
@test c.coefs ≈ coefs

# Standardized from R lars
lambdas = [30.4807287086601,27.6335762471999,24.5043850474775,12.0241681293273,6.37199303756002,0.94518638046917,0.260226891648141,0.224461196798777,0.075770527267496,0.0]
coefs = [
 0.0 0.0 0.0 0.352654719859844 1.07847040186453 1.42732851597461 1.76748835906405 1.8863131810134 1.93449689668163 2.6320434640818
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.119358118484912 0.110381263002499 0.868435306433071
 0.0 0.0 0.127201072264406 0.490672181645049 0.0 0.0 -0.400865382593042 -0.504893808532568 -0.567222747740222 -1.15776369085485
 0.0 -0.114717897632236 -0.218167714095885 -0.595087071604407 -0.397066052933906 -0.394026886896636 -0.234451708513766 0.0 0.0 1.47139070850328
 0.0 0.0 0.0 0.0 -0.73784811174737 -1.04500713257238 -1.42521571193422 -1.57739889812771 -1.62772569964171 -2.53248226081674
]

c = lars(X, y)
@test c.lambdas ≈ lambdas
@test c.coefs ≈ coefs

# Unstandardized from scikit-learn
lambdas = [766.1750193137241,542.1829897748706,523.136666260976,374.9244286861718,33.3577875386257,58.72119349256296,325.92271025568454,0.0]
coefs = [
 0.0 0.0 0.03475908385654046 0.22208390493742114 1.7629499769107984 2.4978426462598082 4.3371433464477835 123.26001716674679
 0.0 0.0 0.0 0.0 0.0 0.0 0.0 128.22587525114494
 0.0 0.0 0.0 0.23772230110830994 -0.49993541998798297 -1.3685754490275883 -3.4530140836158307 -104.45343207073232
 0.0 -0.3621673838611561 -0.38689480994549247 -0.5515575656651571 -0.2000726289404373 0.14972220652359275 1.3073191008578129 250.7202635486004
 0.0 0.0 0.0 0.0 -1.4488853824379428 -2.262124187090131 -4.366091225139248 -158.38233970553551
]

c = lars(X, y, method=:lar, standardize=false, intercept=false)
@test c.lambdas ≈ lambdas
@test c.coefs ≈ coefs

@testset "degenerate regressors produces warning" begin

    rng = StableRNG(123)
    n = 100
    p = 5
    X = randn(rng, n, p)
    X[:, 2] = X[:, 1] + 1e-10*randn(n)
    y = X[:, 1] + randn(n)
    @test_logs (:warn, r"^Regressors in active set degenerate.") lars(X, y, method=:lars)

end
