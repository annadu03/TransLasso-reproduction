library(glmnet)
library(mvtnorm)

source("TransLasso-functions.R")
Coef.gen<- function(s, h,q=30, size.A0, M, sig.beta,sig.delta1, sig.delta2, p, exact=T){
  beta0 <- c(rep(sig.beta,s), rep(0, p - s))
  W <- rep.col(beta0,  M)# ten prior estimates
  W[1,]<-W[1,]-2*sig.beta
  for(k in 1:M){
    if(k <= size.A0){
      if(exact){
        samp0<- sample(1:p, h, replace=F)
        W[samp0,k] <-W[samp0,k] + rep(-sig.delta1, h)
      }else{
        W[1:100,k] <-W[1:100,k] + rnorm(100, 0, h/100)
      }
    }else{
      if(exact){
        samp1 <- sample(1:p, q, replace = F)
        W[samp1,k] <- W[samp1,k] + rep(-sig.delta2,q)
      }else{
        W[1:100,k] <-W[1:100,k] + rnorm(100, 0, q/100)
      }
    }
  }
  return(list(beta0 = beta0, W = W))
}


run_one_simulation <- function(h, size.A0) {

    p <- 500
    s <- 16
    M <- 20
    sig.beta <- 0.3

    n0 <- 150
    n.vec <- c(n0, rep(100, M))

    Sig.X <- diag(1, p)
    l1 <- TRUE


    # -------------------------
    # Generate coefficients
    # -------------------------
    coef.all <-Coef.gen( s, h = h, q = 2*s, size.A0 = size.A0,  M = M,   sig.beta = sig.beta,
             sig.delta1 = sig.beta, sig.delta2 = sig.beta+0.2, p = p, exact=TRUE)
    B <- cbind(coef.all$beta0, coef.all$W)
    beta0 <- coef.all$beta0


    # -------------------------
    # Generate data
    # -------------------------

    X <- NULL
    y <- NULL
    for (k in 1:(M + 1)) {
    X <- rbind(X, rmvnorm(n.vec[k], rep(0, p), Sig.X))
    ind.k <- ind.set(n.vec, k)
    y <- c(y, X[ind.k, ] %*% B[, k] + rnorm (n.vec[k], 0, 1))
    }
    mse.vec <- rep(NA,5)


    # -------------------------
    # 1. Target-only Lasso
    # -------------------------
    beta.init <-
    as.numeric(glmnet(X[1:n.vec[1], ], y[1:n.vec[1]], lambda = sqrt(2 * log(p) / n.vec[1]))$beta)
    mse.vec[1] = mse.fun(as.numeric(beta.init), beta0)$est.err

    # -------------------------
    # 2. Oracle Trans-Lasso
    # -------------------------
    if (size.A0 == 0) {
    beta.kA <- beta.init
    } else{
    beta.kA <- las.kA(X, y, A0 = 1:size.A0, n.vec = n.vec, l1=l1)$beta.kA
    }
    mse.vec[2] = mse.fun(as.numeric(beta.kA), beta0)$est.err

    # -------------------------
    # 3. Trans-Lasso
    # -------------------------
    prop.re1 <- Trans.lasso(X, y, n.vec, I.til = 1:50, l1 = l1)
    prop.re2 <- Trans.lasso(X, y, n.vec, I.til = 101:n.vec[1], l1=l1)

    beta.prop <- (prop.re1$beta.hat + prop.re2$beta.hat) / 2
    mse.vec[3] = mse.fun(beta.prop, beta0)$est.err

    # -------------------------
    # 4. Ad hoc l1-transfer
    # -------------------------
    prop.sp.re1 <- Trans.lasso.sp(X, y, n.vec, I.til = 1:50, l1 = l1)
    prop.sp.re2 <- Trans.lasso.sp(X, y, n.vec, I.til = 101:n.vec[1], l1=l1)
    beta.sp <- (prop.sp.re1$beta.sp + prop.sp.re2$beta.sp) / 2
    mse.vec[4] = mse.fun(beta.sp, beta0)$est.err    
    # -------------------------
    # 5. Agg-Lasso
    # -------------------------
    beta.pool<-(prop.re1$beta.pool+prop.re2$beta.pool)/2
    mse.vec[5] = mse.fun(beta.pool, beta0)$est.err  

    names(mse.vec) <- c(
        "Lasso",
        "Oracle Trans-Lasso",
        "Trans-Lasso",
        "Ad hoc l1-transfer",
        "Agg-Lasso"
    )

    return(mse.vec)
}


# test <- run_one_simulation(
#     h = 6,
#     size.A0 = 12
# )

# print(test)

set.seed(123)

h.grid <- c(2, 6, 12)

A.grid <- c(
    0,
    4,
    8,
    12,
    16,
    20
)

Niter <- 200

results <- data.frame()

for(h in h.grid){

    for(size.A0 in A.grid){

        for(iter in 1:Niter){

            cat(
                "h =", h,
                "| A =", size.A0,
                "| iter =", iter,
                "\n"
            )

            mse <- run_one_simulation(
                h = h,
                size.A0 = size.A0
            )

            temp <- data.frame(
                h = h,
                size.A0 = size.A0,
                iteration = iter,
                method = names(mse),
                SSE = as.numeric(mse)
            )

            results <- rbind(
                results,
                temp
            )
        
        }
        write.csv(
            results,
            "figure1_config1_raw.csv",
            row.names = FALSE
        )
    }
}
library(dplyr)

summary.results <-
    results %>%
    group_by(
        h,
        size.A0,
        method
    ) %>%
    summarise(
        mean_SSE = mean(SSE),
        sd_SSE = sd(SSE),
        se_SSE = sd(SSE)/sqrt(n()),
        .groups = "drop"
    )
library(ggplot2)

ggplot(
    summary.results,
    aes(
        x = size.A0,
        y = mean_SSE,
        group = method,
        linetype = method,
        shape = method
    )
) +
    geom_line() +
    geom_point(size = 2) +
    facet_wrap(
        ~h,
        nrow = 1,
        labeller = label_bquote(
            h == .(h)
        )
    ) +
    scale_x_continuous(
        breaks = c(
            0,4,8,12,16,20
        )
    ) +
    labs(
        x = "|A|",
        y = expression(
            paste(
                "Estimation error  ",
                "||",
                hat(beta),
                "-",
                beta,
                "||"[2]^2
            )
        )
    ) +
    theme_bw()

