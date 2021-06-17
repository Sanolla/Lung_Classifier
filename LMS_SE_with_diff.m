function Output = LMS_SE_with_diff(x1,x2)
n=16;
mu=0.1;
lmsSE = dsp.LMSFilter('Length',n,'StepSize',mu, 'Method', 'Sign-Error LMS');
[~,errdiff]=lmsSE(diff(x2),diff(x1));
% [~,Output]=lmsSE(x2,x1);

Output=filter([1 -1], [1 -(1-1e-3)],cumsum(errdiff));
reset(lmsSE);
end

