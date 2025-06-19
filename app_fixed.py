            # Style the table with differences and bold totals
            def style_diff_and_total(df):
                # Create an empty DataFrame of strings with the same shape as our data
                styles = pd.DataFrame(index=df.index, columns=df.columns, data='')
                
                # Add yellow background to differences (except in total row)
                for idx in df.index:
                    if idx != 'Total Flavour':
                        for col in df.columns:
                            if col in curr_inv_table.columns and col != 'Total Mg':
                                try:
                                    if float(df.loc[idx, col]) != float(curr_inv_table.loc[idx, col]):
                                        styles.loc[idx, col] = 'background-color: yellow'
                                except:
                                    pass
                
                # Make Total Flavour row and Total Mg column bold with larger font and distinct styling
                total_style = 'font-weight: 900; font-size: 20px; background-color: #2C3E50; color: white; border: 2px solid #ECF0F1;'
                for col in df.columns:
                    styles.loc['Total Flavour', col] = total_style
                    if col == 'Total Mg':
                        for idx in df.index:
                            current_style = styles.loc[idx, col]
                            styles.loc[idx, col] = current_style + '; ' + total_style if current_style else total_style
                
                return styles 