from mosek.fusion import Domain, Expr, Model, ObjectiveSense, Var


def mosek_efficient_frontier(n,
                             mu,
                             GT,
                             alpha_value,
                             ind_dist,
                             ind_bound,
                             style_expo,
                             style_bound,
                             idx_wgt,
                             set0_index,
                             wgt_deviation,
                             isin_idx,
                             wgt_in_idx_budget
                             ):
    with Model("Efficient frontier") as M:
        # Defines the variables (holdings). Shortselling is not allowed.
        x = M.variable("x", n, Domain.greaterThan(0.0))  # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())  # Variance variable

        # Total budget constraint
        M.constraint('budget', Expr.sum(x), Domain.equalsTo(1))

        # # Computes the risk
        M.constraint('variance', Expr.vstack(s, 0.5, Expr.mul(GT, Expr.sub(x, idx_wgt))), Domain.inRotatedQCone())

        M.constraint('weight deviation-', Expr.sub(Expr.sub(x, idx_wgt),
                                                   -wgt_deviation
                                                   ), Domain.greaterThan(0))

        M.constraint('weight deviation', Expr.sub(Expr.sub(x, idx_wgt),
                                                  wgt_deviation
                                                  ), Domain.lessThan(0))
        # # Industry distribution
        M.constraint('industry-', Expr.sub(Expr.mul(ind_dist, Expr.sub(x, idx_wgt)),
                                           -ind_bound
                                           ), Domain.greaterThan(0))

        M.constraint('industry', Expr.sub(Expr.mul(ind_dist, Expr.sub(x, idx_wgt)),
                                          ind_bound
                                          ), Domain.lessThan(0))
        # # Style exposure
        # M.constraint('style-', Expr.sub(Expr.mul(style_expo,
        #                                          Expr.sub(x, idx_wgt)),
        #                                 -style_bound
        #                                 ), Domain.greaterThan(0))

        M.constraint('style', Expr.sub(Expr.mul(style_expo,
                                                Expr.sub(x, idx_wgt)),
                                       style_bound
                                       ), Domain.lessThan(0))
        M.constraint('isin_idx_budget', Expr.sub(Expr.sum(Expr.mulElm(isin_idx, x)),
                                                 wgt_in_idx_budget), Domain.greaterThan(0))
        # Bench index members not in universe
        M.constraint('ST', Expr.mulElm(x, set0_index), Domain.equalsTo(0))
        # Define objective as a weighted combination of return and variance
        alpha = M.parameter()
        M.objective('obj', ObjectiveSense.Maximize, Expr.sub(Expr.dot(mu, Expr.sub(x, idx_wgt)), Expr.mul(alpha, s))) # Expr.sub(Expr.dot(mu, x), Expr.mul(alpha, s))

        # Solve multiple instances by varying the parameter alpha
        alpha.setValue(alpha_value)
        M.solve()

        return x.level()
