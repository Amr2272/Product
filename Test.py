# Display real-time result
        if st.session_state.real_time_predictions:
            latest = st.session_state.real_time_predictions[0]
            
            st.markdown("### 📊 Latest Prediction")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Date", latest['date'].strftime('%Y-%m-%d'))
            with col2:
                st.metric("Prediction", f"{latest['prediction']:,.0f}")
            with col3:
                st.metric("Lower Bound", f"{latest['lower_bound']:,.0f}")
            with col4:
                st.metric("Upper Bound", f"{latest['upper_bound']:,.0f}")
            
            # Visualization for real-time prediction
            st.markdown("### 📈 Real-Time Prediction Visualization")
            
            fig_rt = go.Figure()
            
            # Plot historical data
            fig_rt.add_trace(go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['y'],
                mode='markers',
                name='Historical Data (Actual)',
                marker=dict(color='blue', size=4)
            ))
            
            # Plot the single prediction point
            fig_rt.add_trace(go.Scatter(
                x=[pd.to_datetime(latest['date'])],
                y=[latest['prediction']],
                mode='markers',
                name='Real-Time Prediction',
                marker=dict(color='red', size=15, symbol='star')
            ))
            
            # Add confidence interval for the prediction point
            fig_rt.add_trace(go.Scatter(
                x=[pd.to_datetime(latest['date']), pd.to_datetime(latest['date'])],
                y=[latest['lower_bound'], latest['upper_bound']],
                mode='lines',
                name='80% Confidence Interval',
                line=dict(color='rgba(255, 0, 0, 0.3)', width=8),
                showlegend=True
            ))
            
            fig_rt.update_layout(
                title=f'Real-Time Prediction for {latest["date"].strftime("%Y-%m-%d")}',
                xaxis_title='Date',
                yaxis_title='Value',
                title_font_size=20,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_rt, use_container_width=True)
            
            # Show JSON response (API simulation)
            with st.expander("🔍 View API Response (JSON)"):
                st.json(latest)
            
            # History of predictions
            if len(st.session_state.real_time_predictions) > 1:
                st.markdown("### 📜 Prediction History")
                history_df = pd.DataFrame(st.session_state.real_time_predictions)
                history_df['date'] = pd.to_datetime(history_df['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(
                    history_df[['date', 'prediction', 'lower_bound', 'upper_bound']],
                    width='stretch'
                )
