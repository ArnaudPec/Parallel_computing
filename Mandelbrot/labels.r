# Copyright 2015 Nicolas Melot
#
# This file is part of Freja.
#
# Freja is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Freja is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Freja. If not, see <http://www.gnu.org/licenses/>.
#


apply_labels = function(data.frame, labels.list = labels)
{  
  for(col in names(labels.list)[names(labels.list) != "columns"])
  {
    ## Only apply the transformation if the column exists, otherwise skip the column
    if(col %in% names(data.frame))
    {
      ## If the corresponding column in the data frame is not a factor, then make it one
      if(!is.factor(data.frame[,col]))
      {
        data.frame[,col] <- as.factor(data.frame[,col])
      }
      
      ## Get levels and respective labels as from the label list
      levels = names(labels.list[[col]])
      labels = as.character(labels.list[[col]])
      
      ## Pad missing labels with values as they appear in data
      if(!all(levels(data.frame[,col]) %in% names(labels.list[[col]])))
      {
        ## Levels of data.frame, where levels that appear in labels appear first and ordered as in the list
        levels = unique(c(names(labels.list[[col]]), levels(data.frame[,col])))
        #levels = names(factor(data.frame[,col], levels=unique(c(labels.list[[col]], levels(data.frame[,col])))))
        
        ## Concatenate labels with Levels of data.frame that are not in the list of labels
        labels = c(as.character(labels.list[[col]]), levels(factor(data.frame[,col][!data.frame[,col] %in% names(labels.list[[col]])])))
      }

      ## Reorder levels of factors in data.frame, and give them labels
      data.frame[,col] <- factor(data.frame[,col], levels=levels, labels=labels)
      
      ## Only keep factors actually used in the table
      data.frame[,col] <- factor(data.frame[,col])
    }
  }
  
  return(data.frame)
}

label = function(col, columns="columns", labels.list = labels)
{
  if(col %in% names(labels.list[[columns]]))
    return(labels.list[[columns]][col])
  else
    return(col)
}

labels = list(
	nb_thread = c("0" = "Seq.", "1" = "Over."),
	loadbalance = c("0" = "Naive", "1" = "Load-balanced (mandatory)", "2" = "Load-balanced (optional)"),
	thread = c("0" = "Thread 1 (seq.)", "1" = "Thread 1", "2" = "Thread 2", "3" = "Thread 3", "4" = "Thread 4", "5" = "Thread 5", "6" = "Thread 6", "7" = "Thread 7", "8" = "Thread 8"),
	columns = c("nb_thread" = "Number of threads")
)
